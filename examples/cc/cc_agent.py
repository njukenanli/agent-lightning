import asyncio
import json
import os
import platform
import time
from typing import Any, Dict, List, Literal, Optional

import yaml

if platform.system() == "Linux":
    import resource

from datasets import Dataset
from swebench.harness.utils import load_swebench_dataset
from transformers import AutoTokenizer as AutoProcessor
from utils.claude_code_controller import ClaudeController
from utils.custom_adapter import LlmProxyTraceToAugmentedTriplet
from utils.custom_callbacks import AddSamplingParams, AddLogprobs
from utils.evaluation import evaluate
from utils.logger import logger
from utils.type import AgentResult, ClaudeCodeStep
from utils.reward import RewardEstimatorWholeSlice

from agentlightning import (
    InMemoryLightningStore,
    LightningStoreServer,
    LitAgentRunner,
    OtelTracer,
    configure_logger,
)
from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.types import LLM, AttemptedRollout, NamedResources, ProxyLLM, Rollout, RolloutRawResult


def load_dataset(path: str = "swe_debug.jsonl", epoch: int = 0, limit: Optional[int] = None) -> Dict[str, Any]:
    instances = []
    with open(path) as f:
        for line in f:
            instance = json.loads(line)
            instance["epoch"] = epoch
            instances.append(instance)

    if limit is not None:
        instances = instances[:limit]
    return instances


class CodingAgent(LitAgent):
    def __init__(
        self,
        namespace: Literal["swebench", "starryzhang"] = "swebench",
        full_set: Literal["princeton-nlp/SWE-bench", "SWE-bench-Live/SWE-bench-Live"] = "princeton-nlp/SWE-bench",
        split: str = "test",
        max_step: int = 5,
        run_method: Literal["python", "cli"] = "cli",
        tools: list[str] = ["Glob", "Grep", "Bash", "Read", "Edit", "Write", "TodoWrite", "WebFetch", "ExitPlanMode"],
        user_prompt: str = "{description}",
        open_file_limit: int = 4096,
        cache_level: str = "env",  # ["none", "base", "env", "instance"]
        clean: bool = False,
        force_rebuild: bool = False,
        timeout: int = 1_800,  # in sec
        instance_image_tag: str = "latest",
    ) -> None:
        super().__init__()
        self.namespace = namespace
        self.full_set = full_set
        self.split = split
        self.max_step = max_step
        self.run_method = run_method

        self.cache_level = cache_level
        self.clean = clean
        self.force_rebuild = force_rebuild
        self.timeout = timeout
        self.instance_image_tag = instance_image_tag

        full_dataset = load_swebench_dataset(full_set, split)
        self.dataset = {each["instance_id"]: each for each in full_dataset}

        self.tools = tools
        self.user_prompt = user_prompt

        # run instances locally
        if platform.system() == "Linux":
            resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    async def rollout_async(
        self, task: Dict[str, Any], resources: NamedResources, rollout: Rollout
    ) -> RolloutRawResult:
        run_id = f"epoch_{task.get('epoch', 0)}"
        image = f"{self.namespace}/sweb.eval.x86_64.{task['instance_id'].lower()}".replace("__", "_1776_")
        reward = 0.0

        llm = resources.get("llm")
        assert llm is not None, "LLM resource is required for rollout."

        llm = self._strip_proxy_helper(llm, rollout)
        # 1. init container
        controller = ClaudeController(
            image,
            task,
            run_id,
            set(self.tools),
            self.user_prompt,
            llm.endpoint,
            llm.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "dummy"),
        )

        try:
            # 2. execute task
            prediction: AgentResult = controller.run_instance(task, max_step=self.max_step, run_method=self.run_method)
            logger(run_id, task["instance_id"], json.dumps(prediction, indent=4))

            # 3. obtain rewards (evaluation result)
            # empty patch
            if (prediction["model_patch"] is None) or (not prediction["model_patch"].strip()):
                intermediate_reward: RewardEstimatorWholeSlice.ReturnType = RewardEstimatorWholeSlice.intermediate_reward(
                    prediction,
                    controller.container,
                    task["patch"],
                    task.get('epoch', 0)
                )
                # todo: how to send intermediate_reward to GPU side?
                del controller
                return reward

            instance_id = prediction["instance_id"]

            result = evaluate(
                prediction,
                self.dataset[instance_id],
                self.cache_level,
                self.clean,
                self.force_rebuild,
                run_id,
                self.timeout,
                namespace=self.namespace,
                instance_image_tag=self.instance_image_tag,
            )

            # error patch
            if result is None:
                intermediate_reward: RewardEstimatorWholeSlice.ReturnType = RewardEstimatorWholeSlice.intermediate_reward(
                    prediction,
                    controller.container,
                    task["patch"],
                    task.get('epoch', 0)
                )
                del controller
                return reward

            report = result[1]
            # resolved/unresolved patch
            if report[instance_id]["resolved"]:
                reward = 1.0
                prediction["success"] = 1.0
            
            intermediate_reward: RewardEstimatorWholeSlice.ReturnType = RewardEstimatorWholeSlice.intermediate_reward(
                prediction,
                controller.container,
                task["patch"],
                task.get('epoch', 0)
            )
            # todo: how to send intermediate_reward to GPU side?
            del controller
            return reward
        except Exception as e:
            logger(run_id, task["instance_id"], f"Exception during rollout: {e}")
            del controller # anyway, resource should be released...
            return reward

    def _strip_proxy_helper(self, proxy_llm: LLM, rollout: Rollout) -> LLM:
        """Convert [`ProxyLLM`][agentlightning.ProxyLLM] instances into concrete LLMs.

        It resolves ProxyLLM instances to their concrete LLM implementation
        by attaching the attempted rollout context. This is only used when the function
        signature accepts an `llm` parameter and strip_proxy is True.

        Args:
            proxy_llm: Candidate LLM resource.
            rollout: Rollout metadata that provides rollout and attempt identifiers.

        Returns:
            [`LLM`][agentlightning.LLM] with rollout context baked into the endpoint.

        Raises:
            ValueError: If the rollout is not an
                [`AttemptedRollout`][agentlightning.AttemptedRollout].
        """

        if not isinstance(proxy_llm, ProxyLLM):
            # Not a ProxyLLM, nothing to strip here.
            return proxy_llm

        # Rollout is still a Rollout here because API is not stabilized yet.
        # In practice, it must be an AttemptedRollout.
        if not isinstance(rollout, AttemptedRollout):
            raise ValueError("Rollout is not an AttemptedRollout.")

        return proxy_llm.with_attempted_rollout(rollout)


def flatten_messages(messages: List[Any]) -> List[Dict[str, str]]:
    flattened: List[Dict[str, str]] = []
    for msg in messages:
        if msg["role"] in ["system", "user"] and isinstance(msg["content"], list):
            msg_content: List[str] = []
            for content in msg["content"]:
                msg_content.append(content["text"])

            msg["content"] = "".join(msg_content)
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            # NOTE:
            # Tool calls are list of dict, though in most case only one tool call is made per call
            # We serialize it as json string here to avoid nested structure
            msg["tool_calls"] = json.dumps(msg["tool_calls"])

        for k in msg:
            assert isinstance(msg[k], str), f"\n>>> {msg}"
        flattened.append(msg)
    return flattened


async def cc_agent_dry_run_sample(
    model_path,
    server_address,
    sonnet_name,
    haiku_name,
    output_dir,
    config,
) -> None:
    """Run a dry run of the cc agent on a single sample.

    This is a simple test function that runs the math agent on the first 4 problems
    using a single worker. Useful for testing the setup and configuration.
    """

    dataset = load_dataset(config["dataset"]["dataset_path"], limit=4)
    tokenizer = AutoProcessor.from_pretrained(model_path)

    logging = configure_logger(name="Claude Code Agent")

    tracer = OtelTracer()
    runner = LitAgentRunner(tracer)
    adapter = LlmProxyTraceToAugmentedTriplet()
    store = LightningStoreServer(InMemoryLightningStore(), host="0.0.0.0", port=7654)
    llm_proxy = LLMProxy(
        port=12358, store=store, callbacks=["return_token_ids", "opentelemetry", AddLogprobs, AddSamplingParams]
    )

    await store.start()

    llm_proxy.update_model_list(
        [
            ModelConfig(
                model_name=f"{sonnet_name}",
                litellm_params={
                    "model": f"hosted_vllm/{model_path}",
                    "api_base": server_address,
                },
            ),
            ModelConfig(
                model_name=f"{haiku_name}",
                litellm_params={
                    "model": f"hosted_vllm/{model_path}",
                    "api_base": server_address,
                },
            ),
        ]
    )
    await llm_proxy.restart()

    # Put the LLM proxy address into the store as an address
    await store.add_resources(
        {
            "llm": llm_proxy.as_resource(model="local"),
        }
    )

    agent = CodingAgent(
        namespace=config["dataset"]["namespace"],
        full_set=config["dataset"]["full_set"],
        split=config["dataset"]["split"],
        max_step=config["runtime"]["max_step"],
        run_method=config["runtime"]["run_method"],
        tools=config["agent"]["tools"],
        user_prompt=config["agent"]["user_prompt"],
    )

    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(
            dataset[0],
        )

        spans = await store.query_spans(rollout.rollout_id)
        triplets = adapter.adapt(spans)
        logging.info(f"dump {len(spans)} spans, extract {len(triplets)} triplets")
        if output_dir is not None:
            with open(
                os.path.join(output_dir, f"stream_{dataset[0]['instance_id']}-{rollout.attempt.attempt_id}.json"), "w"
            ) as f:
                for span in spans:
                    f.write(json.dumps(span.model_dump()) + "\n")

            all_triplets: List[Dict[str, Any]] = []
            recent_reward: Optional[float] = None
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    recent_reward = triplet.reward

                prompt = tokenizer.decode(triplet.prompt["token_ids"])  # type: ignore
                all_triplets.append(
                    {
                        "repo": rollout.input["repo"],
                        "instance_id": rollout.input["instance_id"],
                        "turn": triplet.metadata["sequence_id"],
                        "prompt_ids": triplet.prompt["token_ids"],
                        "gold_completion_ids": triplet.response["token_ids"],
                        "logprobs": triplet.response["logprobs"],
                        "reward": recent_reward,
                        "prompt": prompt,
                        "messages": flatten_messages(triplet.metadata["messages"]),
                    }
                )

            ds = Dataset.from_list(all_triplets)
            ds.save_to_disk(os.path.join(output_dir, f"dataset-{dataset[0]['instance_id']}"))
            logging.info(f"Saved dataset with {len(ds)} samples to dataset-{dataset[0]['instance_id']}")

    await llm_proxy.stop()


async def gold_cc_agent_run_dataset(
    sonnet_name,
    haiku_name,
    output_dir,
    config,
):
    """Run a dry run of the cc agent on a single sample.

    This is a simple test function that runs the math agent on the first 4 problems
    using a single worker. Useful for testing the setup and configuration.
    """
    dataset = load_dataset(config["dataset"]["dataset_path"])

    logging = configure_logger(name="Claude Code Agent")

    tracer = OtelTracer()
    runner = LitAgentRunner(tracer)
    store = LightningStoreServer(InMemoryLightningStore(), host="0.0.0.0", port=7654)
    llm_proxy = LLMProxy(
        port=12358,
        store=store,
        callbacks=[
            "opentelemetry",
        ],
    )

    await store.start()

    llm_proxy.update_model_list(
        [
            ModelConfig(
                model_name=f"{sonnet_name}",
                litellm_params={"model": f"anthropic/{sonnet_name}", "api_key": "os.environ/ANTHROPIC_API_KEY"},
            ),
            ModelConfig(
                model_name=f"{haiku_name}",
                litellm_params={"model": f"anthropic/{haiku_name}", "api_key": "os.environ/ANTHROPIC_API_KEY"},
            ),
        ]
    )
    await llm_proxy.restart()

    # Put the LLM proxy address into the store as an address
    await store.add_resources(
        {
            "llm": llm_proxy.as_resource(model="local"),
        }
    )

    for each in dataset:
        agent = CodingAgent(
            namespace=config["dataset"]["namespace"],
            full_set=config["dataset"]["full_set"],
            split=config["dataset"]["split"],
            max_step=config["runtime"]["max_step"],
            run_method=config["runtime"]["run_method"],
            tools=config["agent"]["tools"],
            user_prompt=config["agent"]["user_prompt"],
        )
        with runner.run_context(agent=agent, store=store):
            rollout = await runner.step(each)
            spans = await store.query_spans(rollout.rollout_id)

        if output_dir is None:
            logging.info(f"instance {each['instance_id']} generate {len(spans)} spans")
        else:
            logging.info(f"instance {each['instance_id']} dump {len(spans)} spans to {output_dir}")
            with open(os.path.join(output_dir, f"{each['instance_id']}.json"), "w") as f:
                for span in spans:
                    f.write(json.dumps(span.model_dump()) + "\n")

        time.sleep(2 * 60)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # extract spans from official Claude Code
    parser.add_argument("--official", action="store_true", help="Whether to run official claude code.")

    # extract spans from hosted LLM server via litellm proxy
    parser.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help="Model name or path."
    )
    parser.add_argument("--server_address", type=str, default="http://localhost:8000/v1", help="LLM server address.")

    # common setup
    parser.add_argument(
        "--sonnet_name", type=str, default="claude-sonnet-4-5-20250929", help="Name of the sonnet model."
    )
    parser.add_argument("--haiku_name", type=str, default="claude-haiku-4-5-20251001", help="Name of the haiku model.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save output logs.")
    parser.add_argument("--agent_config", type=str, default="agent_config.yaml", help="Configs to run claude code.")

    args = parser.parse_args()

    with open(args.agent_config) as f:
        config = yaml.safe_load(f)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if not args.official:
        asyncio.run(
            cc_agent_dry_run_sample(
                model_path=args.model_name_or_path,
                server_address=args.server_address,
                sonnet_name=args.sonnet_name,
                haiku_name=args.haiku_name,
                output_dir=args.output_dir,
                config=config,
            )
        )
    else:
        asyncio.run(
            gold_cc_agent_run_dataset(
                sonnet_name=args.sonnet_name,
                haiku_name=args.haiku_name,
                output_dir=args.output_dir,
                config=config,
            )
        )
