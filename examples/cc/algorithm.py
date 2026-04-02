import asyncio
import json
import math
import os
import random
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import httpx
from cc_agent import flatten_messages, load_dataset
from datasets import Dataset, DatasetDict
from rich.console import Console
from transformers import AutoTokenizer
from utils.custom_adapter import LlmProxyTraceToAugmentedTriplet
from utils.custom_callbacks import AddLogprobs

from agentlightning import configure_logger
from agentlightning.adapter import LlmProxyTraceToTriplet
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.store import LightningStore, LightningStoreClient
from agentlightning.types import Rollout

console = Console()


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@contextmanager
def vllm_server(
    model_path: str,
    port: int,
    vllm_serve_args_str: str,
    enable_lora: bool = False,
    startup_timeout: float = 300.0,
    terminate_timeout: float = 10.0,
):
    """Serves a vLLM model from command line.

    Args:
        model_path: The path to the vLLM model. It can be either a local path or a Hugging Face model ID.
        port: The port to serve the model on.
        startup_timeout: The timeout for the server to start.
        terminate_timeout: The timeout for the server to terminate.
        max_model_len: The maximum model length.
        gpu_memory_utilization: The GPU memory utilization for the server. Set it lower to avoid OOM.
        quantization: The quantization method.
        auto_tool_choice: Whether to enable auto tool choice.
        tool_call_parser: The tool call parser to use.
    """
    proc: Optional[subprocess.Popen[bytes]] = None
    try:
        vllm_serve_args = [
            "--port",
            str(port),
        ]
        if vllm_serve_args_str:
            vllm_serve_args.extend(vllm_serve_args_str.strip().split())

        if enable_lora:
            vllm_serve_args.extend(["--enable-lora"])

        proc = subprocess.Popen(["vllm", "serve", model_path, *vllm_serve_args])

        # Wait for the server to be ready
        url = f"http://localhost:{port}/health"
        start = time.time()
        client = httpx.Client()

        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                result = proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None
                time.sleep(0.5)
                if time.time() - start > startup_timeout:
                    raise RuntimeError(f"Server failed to start in {startup_timeout} seconds.") from None

        yield f"http://localhost:{port}/v1"
    finally:
        # Terminate the server
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(terminate_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


async def run_rollout(
    model_path: str,
    vllm_serve_args_str: str,
    llm_proxy: LLMProxy,
    store: LightningStore,
    task_dataset: Any,
    num_repeats: int = 1,
    enable_lora: bool = False,
) -> List[Rollout]:
    """Rollout to get trace data"""
    with vllm_server(model_path, _find_available_port(), vllm_serve_args_str, enable_lora) as server_address:
        llm_proxy.update_model_list(
            [
                ModelConfig(
                    model_name="claude-sonnet-4-5-20250929",
                    litellm_params={
                        "model": f"hosted_vllm/{model_path}",
                        "api_base": server_address,
                    },
                ),
                ModelConfig(
                    model_name="claude-haiku-4-5-20251001",
                    litellm_params={
                        "model": f"hosted_vllm/{model_path}",
                        "api_base": server_address,
                    },
                ),
            ]
        )

        await llm_proxy.restart()

        resources_update = await store.add_resources({"llm": llm_proxy.as_resource(model="local")})

        rollouts: List[Rollout] = []
        for _ in range(num_repeats):
            for data in task_dataset:
                rollouts.append(
                    await store.enqueue_rollout(input=data, mode="train", resources_id=resources_update.resources_id)
                )

        console.print(f"[bold red][Algo][/bold red] Enqueued {len(rollouts)} rollouts")

        # Wait for the tasks to complete
        completed_rollouts: List[Rollout] = []

        while True:
            completed_rollouts = await store.wait_for_rollouts(
                rollout_ids=[rollout.rollout_id for rollout in rollouts],
                timeout=0.0,  # Timeout must be a very small value to avoid blocking the store server
            )
            if len(completed_rollouts) >= len(rollouts):
                console.print(f"[bold red][Algo][/bold red] Received all {len(rollouts)} rollouts")
                break
            console.print(
                f"[bold red][Algo][/bold red] Received {len(completed_rollouts)} rollouts, waiting for more..."
            )
            await asyncio.sleep(5.0)

        await llm_proxy.stop()
        return completed_rollouts


async def build_dataset(
    completed_rollouts: List[Rollout],
    store: LightningStore,
    data_adapter: LlmProxyTraceToTriplet,
    tokenizer: AutoTokenizer,
    epoch: int,
    train_triplet_fraction: float,
    dataset_dump_path: str,
    span_dump_path: Optional[str] = None,
) -> str:
    """Prepare the dataset for training"""
    all_triplets: List[Dict[str, Any]] = []

    spand_dump_epoch_path = None
    if span_dump_path:
        spand_dump_epoch_path = os.path.join(span_dump_path, f"epoch_{epoch}")
        os.makedirs(spand_dump_epoch_path, exist_ok=True)

    task_result = []
    for rollout in completed_rollouts:
        # Use data_adapter to adapt the spans to triplets. Triplets are a list of Pydantic models:
        spans = await store.query_spans(rollout.rollout_id)
        triplets = data_adapter.adapt(spans)

        # Logging the prompt and response lengths and rewards for debugging
        prompt_lengths = [len(t.prompt["token_ids"]) if t.prompt["token_ids"] else 0 for t in triplets]
        response_lengths = [len(t.response["token_ids"]) if t.response["token_ids"] else 0 for t in triplets]
        console.print(
            f"[bold red][Algo][/bold red] Rollout {rollout.rollout_id} has {len(spans)} spans, yielding {len(triplets)} triplets. "
            f"Prompt lengths: {prompt_lengths}. Response lengths: {response_lengths}. "
            f"Rewards are: {[t.reward for t in triplets]}"
        )

        if triplets and triplets[-1].reward is not None:
            task_result.append(triplets[-1].reward)

        # Converts the triplets to a HuggingFace Dataset
        # NOTE:
        # - multiprocesses to speed up;
        # - maybe customize reward for each triple in the future
        recent_reward: Optional[float] = None
        for triplet in reversed(triplets):
            if triplet.reward is not None:
                recent_reward = triplet.reward

            if recent_reward is None:
                console.print(
                    f"[bold red][Algo][/bold red] Recent reward is None for triplet {triplet}. "
                    "Skip adding to SFT training data."
                )
                continue

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

        if spand_dump_epoch_path:
            span_file_path = os.path.join(
                spand_dump_epoch_path, f"{rollout.input['instance_id']}-{rollout.rollout_id}.json"
            )
            with open(span_file_path, "w") as f:
                for span in spans:
                    f.write(json.dumps(span.model_dump()) + "\n")

    if len(task_result) < len(completed_rollouts):
        console.print(
            f"[bold red][Algo][/bold red] Warning: only {len(task_result)} rewards collected from "
            f"{len(completed_rollouts)} rollouts."
        )
    console.print(
        f"[bold red][Algo][/bold red] Epoch {epoch} averaged reward {sum(task_result)/len(completed_rollouts) if completed_rollouts else 0.0} in {len(completed_rollouts)} rollouts."
    )

    if len(all_triplets) == 0:
        raise ValueError("No triplets to train on.")

    # NOTE:
    # Here we do not handle data leakage where training samples are newer than eval set
    random.shuffle(all_triplets)
    split_point = math.ceil(len(all_triplets) * train_triplet_fraction)
    assert split_point < len(all_triplets) - 1
    train_triplets = all_triplets[:split_point]
    eval_triplets = all_triplets[split_point:]

    train_dataset = Dataset.from_list(train_triplets)  # type: ignore
    eval_dataset = Dataset.from_list(eval_triplets) if eval_triplets is not None else None  # type: ignore
    console.print(
        f"[bold red][Algo][/bold red] Generated {len(all_triplets)} triplets for SFT training. "
        f"Keeping {len(train_triplets)} for training."
    )

    dataset_path = os.path.join(dataset_dump_path, f"epoch_{epoch}")
    DatasetDict({"train": train_dataset, "test": eval_dataset}).save_to_disk(  # type: ignore
        dataset_path
    )  # type: ignore
    return dataset_path


def run_trainer(config_path: str, model_path: str, dataset_path: str, output_dir: str) -> None:
    """Start training and save the model"""
    # Launch the training script via accelerate
    cmd = ["accelerate", "launch", "--config_file", config_path, "offline_rloo.py"]

    # script arguments
    cmd += [
        "--model_name_or_path",
        model_path,
        "--dataset_dir",
        dataset_path,
        "--output_dir",
        output_dir,
    ]

    console.print(f"[bold red][Algo][/bold red] {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False, text=True)

    exit_code = result.returncode

    if exit_code != 0:
        raise RuntimeError(f"Training process exited with code {exit_code}")


async def run_epoch(
    epoch: int,
    store: LightningStore,
    model_path: str,
    tokenizer: AutoTokenizer,
    vllm_serve_args_str: str,
    task_dataset: Any,
    num_repeats: int,
    llm_proxy: LLMProxy,
    data_adapter: LlmProxyTraceToTriplet,
    train_triplet_fraction: float,
    dataset_dump_path: str,
    config_path: str,
    model_output_dir: str = "models",
) -> str:
    console.print(f"\n[bold red][Algo][/bold red] Starting epoch {epoch}")

    # 1. Run rollout to get traces
    completed_rollouts = await run_rollout(
        model_path=model_path,
        vllm_serve_args_str=vllm_serve_args_str,
        llm_proxy=llm_proxy,
        store=store,
        task_dataset=task_dataset,
        num_repeats=num_repeats,
        enable_lora=epoch > 0,
    )

    # 2. Build dataset from the traces
    dataset_path = await build_dataset(
        completed_rollouts=completed_rollouts,
        store=store,
        data_adapter=data_adapter,
        tokenizer=tokenizer,
        epoch=epoch,
        train_triplet_fraction=train_triplet_fraction,
        dataset_dump_path=dataset_dump_path,
    )

    # 3. Run trainer to train the model
    next_model_path = os.path.join(model_output_dir, f"version_{epoch+1}")
    run_trainer(config_path, model_path, dataset_path, next_model_path)
    return next_model_path


async def run_algorithm(
    store: LightningStore,
    proxy_port: int,
    access_host: Optional[str],
    model_path: str,
    vllm_serve_args_str: str,
    num_epochs: int,
    num_repeats: int,
    train_triplet_fraction: float,
    dataset_path: str,
    dataset_dump_path: str,
    config_path: str,
) -> None:
    """An example training algorithm that communicates with rollout runners via the store.

    Args:
        store: The LightningStoreClient instance.
    """

    llm_proxy = LLMProxy(port=proxy_port, store=store, callbacks=["return_token_ids", "opentelemetry", AddLogprobs])
    if access_host is not None:
        llm_proxy.server_launcher.args.access_host = access_host

    tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore

    data_adapter = LlmProxyTraceToAugmentedTriplet()

    for epoch in range(num_epochs):
        task_dataset = load_dataset(dataset_path, epoch=epoch)
        model_path = await run_epoch(
            epoch=epoch,
            store=store,
            model_path=model_path,
            tokenizer=tokenizer,  # type: ignore
            vllm_serve_args_str=vllm_serve_args_str,
            task_dataset=task_dataset,
            num_repeats=num_repeats,
            llm_proxy=llm_proxy,
            data_adapter=data_adapter,
            train_triplet_fraction=train_triplet_fraction,
            dataset_dump_path=dataset_dump_path,
            config_path=config_path,
        )

    console.print(f"[bold red][Algo][/bold red] Final model path: {model_path}")


if __name__ == "__main__":
    configure_logger()
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--store_address", type=str, default="http://localhost:4747", help="The address of the LightningStore server."
    )
    parser.add_argument("--access_host", type=str, default=None, help="The access host for the LLM proxy server.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        help="The model name or path for the LLM backend.",
    )
    parser.add_argument(
        "--vllm_serve_args_str",
        type=str,
        default="--enable-auto-tool-choice --tool-call-parser qwen3_coder",
        help="The additional arguments for vLLM serve.",
    )
    parser.add_argument("--proxy_port", type=int, default=8765, help="The port for the LLM proxy server.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of repeats.")
    parser.add_argument(
        "--train_triplet_fraction",
        type=float,
        default=0.8,
    )
    parser.add_argument("--dataset_path", type=str, default="swe_debug.jsonl")
    parser.add_argument(
        "--dataset_dump_path", type=str, default="datasets", help="If not None, dump the dataset to disk."
    )
    parser.add_argument("--config_path", type=str, default="accelerate_config.yaml")
    args = parser.parse_args()

    store = LightningStoreClient(args.store_address)

    asyncio.run(
        run_algorithm(
            store=store,
            proxy_port=args.proxy_port,
            access_host=args.access_host,
            model_path=args.model_name_or_path,
            vllm_serve_args_str=args.vllm_serve_args_str,
            num_epochs=args.num_epochs,
            num_repeats=args.num_repeats,
            train_triplet_fraction=args.train_triplet_fraction,
            dataset_path=args.dataset_path,
            dataset_dump_path=args.dataset_dump_path,
            config_path=args.config_path,
        )
    )
