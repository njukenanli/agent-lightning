import asyncio
import multiprocessing
from typing import Any, Dict

import yaml
from cc_agent import CodingAgent
from rich.console import Console

from agentlightning import LitAgentRunner, configure_logger
from agentlightning.store import LightningStore, LightningStoreClient
from agentlightning.tracer import OtelTracer
from examples.cc.utils.type import AgentConfig

console = Console()


def run_rollout(*, store: LightningStore, config: AgentConfig, worker_id: int) -> None:
    tracer = OtelTracer()
    runner = LitAgentRunner[Dict[str, Any]](tracer)

    console.print(f"[bold green]Runners: [/bold green] Rollout runner {worker_id} started.")

    agent = CodingAgent(
        namespace=config["dataset"]["namespace"],
        full_set=config["dataset"]["full_set"],
        split=config["dataset"]["split"],
        max_step=config["runtime"]["max_step"],
        run_method=config["runtime"]["run_method"],
        tools=config["agent"]["tools"],
        user_prompt=config["agent"]["user_prompt"],
    )

    with runner.run_context(agent=agent, store=store, worker_id=worker_id):
        asyncio.run(runner.iter())


def spawn_runners(*, store: LightningStore, config: AgentConfig) -> None:
    runners = [
        multiprocessing.Process(target=run_rollout, kwargs={"store": store, "config": config, "worker_id": worker_id})
        for worker_id in range(config["runtime"]["workers"])
    ]
    for runner in runners:
        runner.start()

    for runner in runners:
        runner.join()


if __name__ == "__main__":
    configure_logger()
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--store_address", type=str, default="http://localhost:4747", help="The address of the LightningStore server."
    )
    parser.add_argument(
        "--agent_config", type=str, default="agent_config.yaml", help="Agent config to run Claude Code."
    )

    args = parser.parse_args()

    with open(args.agent_config) as f:
        config = yaml.safe_load(f)

    store = LightningStoreClient(args.store_address)
    spawn_runners(store=store, config=config)
