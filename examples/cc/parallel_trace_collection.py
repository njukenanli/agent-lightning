import asyncio
import time

import yaml
from src.algorithm import build_dataset, run_rollout
from src.cc_agent import load_dataset
from transformers import AutoTokenizer
from src.utils.custom_adapter import LlmProxyTraceToAugmentedTriplet
from src.utils.custom_callbacks import AddSamplingParams, AddLogprobs
from src.utils.response_validation_middleware import (
    ResponseValidationMiddleware,
    StepWarningMiddleware,
    ToolSelectionMiddleware,
    set_allowed_tools,
    set_max_step,
)

from agentlightning.llm_proxy import LLMProxy
from agentlightning.store import LightningStoreClient


async def run_parallel_trace_collection(
    model_path: str,
    vllm_serve_args_str: str,
    llm_proxy: str,
    store,
    task_dataset,
    num_repeats: int,
    data_adapter,
    tokenizer,
    epoch,
    train_triplet_fraction,
    dataset_dump_path,
    span_dump_path,
) -> None:
    # 1. Run rollout to get traces
    start = time.time()
    completed_rollouts = await run_rollout(
        model_path=model_path,
        vllm_serve_args_str=vllm_serve_args_str,
        llm_proxy=llm_proxy,
        store=store,
        task_dataset=task_dataset,
        num_repeats=num_repeats,
        enable_lora=epoch > 0,
    )

    print(f"Trace collection took {time.time() - start} seconds.")
    start = time.time()
    # 2. Build dataset from the traces
    await build_dataset(
        completed_rollouts=completed_rollouts,
        store=store,
        data_adapter=data_adapter,
        tokenizer=tokenizer,
        epoch=epoch,
        train_triplet_fraction=train_triplet_fraction,
        dataset_dump_path=dataset_dump_path,
        span_dump_path=span_dump_path,
    )
    print(f"Dataset building took {time.time() - start} seconds.")


if __name__ == "__main__":
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
    parser.add_argument(
        "--train_triplet_fraction",
        type=float,
        default=0.8,
    )
    parser.add_argument("--dataset_path", type=str, default="swe_debug.jsonl")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of repeats.")
    parser.add_argument(
        "--dataset_dump_path", type=str, default="datasets", help="If not None, dump the dataset to disk."
    )
    parser.add_argument("--span_dump_path", type=str, default="spans", help="If not None, dump the spans to disk.")
    parser.add_argument("--enable_lora", action="store_true", help="Whether to enable LoRA during trace collection.")
    parser.add_argument(
        "--agent_config", type=str, default="agent_config.yaml", help="Agent config to run Claude Code."
    )

    args = parser.parse_args()

    with open(args.agent_config) as f:
        config = yaml.safe_load(f)

    store = LightningStoreClient(args.store_address)

    # Configure middleware settings from the agent config.
    set_allowed_tools(set(config["agent"]["tools"]))
    set_max_step(config["runtime"]["max_step"])

    llm_proxy = LLMProxy(
        port=args.proxy_port,
        store=store,
        callbacks=["return_token_ids", "opentelemetry", AddLogprobs, AddSamplingParams],
        # Custom middlewares are listed first so they are added first and thus run
        # innermost (closest to the backend), seeing raw non-streaming JSON.
        middlewares=[ToolSelectionMiddleware, StepWarningMiddleware, ResponseValidationMiddleware, "rollout_attempt", "stream_conversion"],
    )
    if args.access_host is not None:
        llm_proxy.server_launcher._access_host = args.access_host

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)  # type: ignore

    data_adapter = LlmProxyTraceToAugmentedTriplet()
    task_dataset = load_dataset(args.dataset_path, epoch=1 if args.enable_lora else 0)
    asyncio.run(
        run_parallel_trace_collection(
            model_path=args.model_name_or_path,
            vllm_serve_args_str=args.vllm_serve_args_str,
            llm_proxy=llm_proxy,
            store=store,
            task_dataset=task_dataset,
            num_repeats=args.num_repeats,
            data_adapter=data_adapter,
            tokenizer=tokenizer,
            epoch=1 if args.enable_lora else 0,
            train_triplet_fraction=args.train_triplet_fraction,
            dataset_dump_path=args.dataset_dump_path,
            span_dump_path=args.span_dump_path,
        )
    )
