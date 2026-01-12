import asyncio
import json
import os
import platform
import time
from typing import Any, Dict, List, Literal, Optional

import yaml

if platform.system() == "Linux":
    import resource

from utils.claude_code_controller import ClaudeController
from utils.logger import logger
from utils.type import AgentResult

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
        run_id: str = "default",
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

        self.tools = tools
        self.user_prompt = user_prompt
        self.run_id = run_id

        # run instances locally
        if platform.system() == "Linux":
            resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    async def rollout_async(
        self, task: Dict[str, Any], resources: NamedResources, rollout: Rollout
    ) -> RolloutRawResult:
        image = task["docker_image"]
        reward = 0.0

        llm = resources.get("llm")
        assert llm is not None, "LLM resource is required for rollout."

        llm = self._strip_proxy_helper(llm, rollout)

        try:
            # 1. init container
            controller = ClaudeController(
                image,
                task,
                self.run_id,
                set(self.tools),
                self.user_prompt,
                llm.endpoint,
                llm.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "dummy"),
            )
            # 2. execute task
            prediction: AgentResult = controller.run_instance(task, max_step=self.max_step, run_method=self.run_method)
            logger(self.run_id, task["instance_id"], json.dumps(prediction, indent=4))
            # Under development: Intermediate Reward
            # intermediate_reward_list: list[tuple[ClaudeCodeStep, float]] = controller.calculate_intermediate_rewards_per_slice(task["patch"],  prediction["model_patch"], prediction["reproduction_file"], prediction["trajectory"])
            del controller
        except Exception as e:
            logger(self.run_id, task["instance_id"], f"Exception during rollout: {e}")
            return reward

        os.makedirs(f"patch/{self.run_id}", exist_ok=True)
        with open(f"patch/{self.run_id}/{task['instance_id']}.diff", "w") as f:
            f.write(prediction["model_patch"]) 
        
        return 0.0

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

def find_idle_port() -> tuple[int, int]:
    '''
    Automatically find 2 ports not occupied.
    '''
    import socket
    
    ports = []
    sockets = []
    
    for _ in range(2):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))  # Bind to any available port
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ports.append(sock.getsockname()[1])
        sockets.append(sock)
    
    # Close all sockets to release the ports
    for sock in sockets:
        sock.close()
    
    print("Find two idle ports:", (ports[0], ports[1]))
    return (ports[0], ports[1])

async def gold_cc_agent_run_dataset(
    config,
    model_name,
    api_base,
    api_key
):
    """Run a dry run of the cc agent on a single sample.

    This is a simple test function that runs the math agent on the first 4 problems
    using a single worker. Useful for testing the setup and configuration.
    """
    sonnet_name = "claude-sonnet-4-5-20250929"
    haiku_name = "claude-haiku-4-5-20251001"
    run_id=model_name.replace("/", "_")
    # Use swe_100.jsonl dataset
    dataset = load_dataset(config["dataset"]["dataset_path"],)

    tracer = OtelTracer()
    runner = LitAgentRunner(tracer)
    port1, port2 = find_idle_port()
    store = LightningStoreServer(InMemoryLightningStore(), host="0.0.0.0", port=port1)
    llm_proxy = LLMProxy(
        port=port2,
        store=store,
        callbacks=[
            "opentelemetry",
        ],
    )

    await store.start()

    # Use CloudGPT configuration (same as cc_apo_algo.py)
    #from utils.cloudgpt_aoai import get_openai_token_provider
    #token_provider = get_openai_token_provider()

    for each in dataset:
        if "azure" not in model_name:
            llm_proxy.update_model_list(
                [
                    ModelConfig(
                        model_name=f"{sonnet_name}",
                        litellm_params={
                            "model": model_name,
                            "api_base": api_base,
                            "api_key": api_key,
                        },
                    ),
                    ModelConfig(
                        model_name=f"{haiku_name}",
                        litellm_params={
                            "model": model_name,
                            "api_base": api_base,
                            "api_key": api_key,
                        },
                    ),
                ]
            )
        else:
            from utils.cloudgpt_aoai import get_openai_token_provider
            token_provider = get_openai_token_provider()
            llm_proxy.update_model_list(
                [
                    ModelConfig(
                        model_name=f"{sonnet_name}",
                        litellm_params={
                            "model": model_name,
                            "api_base": "https://cloudgpt-openai.azure-api.net/",
                            "api_version": "2025-04-01-preview",
                            "azure_ad_token": token_provider(),
                        },
                    ),
                    ModelConfig(
                        model_name=f"{haiku_name}",
                        litellm_params={
                            "model": model_name,
                            "api_base": "https://cloudgpt-openai.azure-api.net/",
                            "api_version": "2025-04-01-preview",
                            "azure_ad_token": token_provider(),
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


        if os.path.exists(f"patch/{run_id}/{each['instance_id']}.diff"):
            with open(f"patch/{run_id}/{each['instance_id']}.diff") as f:
                patch=f.read()
        else:
            patch=""
        if os.path.exists(f"logs/{run_id}/{each['instance_id']}"):
            with open(f"logs/{run_id}/{each['instance_id']}") as f:
                lg=f.read()
            if '"num_turns": ' in lg:
                try:
                    turns = lg.split('"num_turns": ')[-1].split(",\n")[0]
                    print(each["instance_id"], turns, "Not a git repository." not in patch)
                    if int(turns) >= 10 and ("Not a git repository." not in patch): # not unexpected exit:
                        print("find valid traj, continue.", flush=True)
                        continue
                except:
                    pass

        agent = CodingAgent(
            namespace=config["dataset"]["namespace"],
            full_set=config["dataset"]["full_set"],
            split=config["dataset"]["split"],
            max_step=config["runtime"]["max_step"],
            run_method=config["runtime"]["run_method"],
            tools=config["agent"]["tools"],
            user_prompt=config["agent"]["user_prompt"],
            run_id=run_id,
        )
        with runner.run_context(agent=agent, store=store):
            rollout = await runner.step(each)
            spans = await store.query_spans(rollout.rollout_id)

        time.sleep(30)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openrouter/google/gemini-3-flash-preview", help="Model name with provider prefix.")
    # or azure/gpt-5.2-20251211
    parser.add_argument("--api_base", type=str, default="https://openrouter.ai/api/v1", help="Model name with provider prefix. For Azure OpenAI this field is not needed.")
    parser.add_argument("--api_key", type=str, default="none", help="API key. For Azure OpenAI we use browser interactive login, so API key is not used.")
    parser.add_argument("--agent_config", type=str, default="agent_config.yaml", help="Configs to run claude code.")
    args = parser.parse_args()

    with open(args.agent_config) as f:
        config = yaml.safe_load(f)

    asyncio.run(
        gold_cc_agent_run_dataset(
            config,
            args.model_name,
            args.api_base,
            args.api_key,
        )
    )
