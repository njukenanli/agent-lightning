import json
from typing import Callable, Literal

import dotenv
from src.utils.docker_runtime import Runtime
from src.utils.reward import RewardEstimatorWholeSlice
from src.utils.type import CC_ALL_TOOLS as all_tools
from src.utils.type import AgentResult, ClaudeCodeStep, ClaudeCodeTraj


class ClaudeController:
    system_prompt = """You are an expert software engineer solving swebench bug fixing tasks."""

    def __init__(
        self, image: str, instance: dict, logger: Callable, tools: set, user_prompt: str, endpoint: str, api_key: str
    ) -> None:
        self.image = image
        self.instance = instance
        self.logger = logger
        self.endpoint = endpoint
        self.api_key = api_key
        self.container: Runtime = self.init_container(self.image, self.instance)
        self.allowed_tools: str = ",".join([f'"{i}"' for i in tools])
        self.disallowed_tools: str = ",".join([f'"{i}"' for i in (all_tools - tools)])
        assert "{description}" in user_prompt
        self.user_prompt: str = user_prompt
        return

    def init_container(self, image: str, instance: dict) -> Runtime:
        container = Runtime.start_session(
            image,
            instance,
            log_function=self.logger
            platform="linux",
        )
        container.send_command("curl -fsSL https://claude.ai/install.sh | bash -s -- 2.0.65")
        container.send_command('alias claude="$HOME/.local/bin/claude"')
        dotenv.load_dotenv()
        # anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        # container.send_command(f"export ANTHROPIC_API_KEY={anthropic_api_key}")
        # if (not os.getenv("ANTHROPIC_BASE_URL")) or (not os.getenv("ANTHROPIC_AUTH_TOKEN")):
        #     raise RuntimeError("ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN not found!")
        container.send_command(f"export ANTHROPIC_BASE_URL={self.endpoint}")
        container.send_command(f"export ANTHROPIC_AUTH_TOKEN={self.api_key}")
        container.send_command("export IS_SANDBOX=1")
        return container

    def _run_cli(self, instance: dict, max_step: int, timelimit: int) -> ClaudeCodeTraj:
        # prepare prompt safely: write it to a file inside the container using a single-quoted heredoc
        # directly applying prompt for heredoc may raise error for windows line ending \r\n
        prompt_text = self.user_prompt.format(description=instance["problem_statement"].replace('"""', "'''"), tools=self.allowed_tools)
        # choose a simple filename and a heredoc delimiter unlikely to collide
        heredoc_cmd = "cat > /tmp/cc_prompt.txt <<'CC_PROMPT'\n" + prompt_text + "\nCC_PROMPT\n"
        self.container.send_command(heredoc_cmd)

        self.container.send_command("mkdir -p /testbed/.claude")
        with open("utils/settings.template.json") as f:
            setting = f.read()
        setting = setting.replace("<allowedTools>", self.allowed_tools).replace(
            "<excludedTools>", self.disallowed_tools
        )
        setting_cmd = "cat > /testbed/.claude/settings.json <<'CC_SETTING'\n" + setting + "\nCC_SETTING\n"
        self.container.send_command(setting_cmd)

        # with open("utils/handle_hook.template.sh") as f:
        #     handler = f.read()
        # handler_cmd = "cat > /tmp/handle_hook.sh <<'CC_HOOK'\n" + handler + "\nCC_HOOK\n"
        # self.container.send_command(handler_cmd)
        # self.container.send_command("chmod +x /tmp/handle_hook.sh")

        # run claude reading the prompt from the file to avoid shell interpolation issues
        claude_cmd = f'claude -p "$(cat /tmp/cc_prompt.txt)" --system-prompt "{self.system_prompt}" --max-turns {max_step}  --output-format json --verbose'
        res = self.container.send_command(claude_cmd, timelimit * 60)
        traj = [i for i in res.output.splitlines() if "session_id" in i]
        assert len(traj) > 0, "traj not found!"
        traj: ClaudeCodeTraj = json.loads(traj[0])
        # self.container.send_command("cat /tmp/hook.out")
        return traj

    def _run_python_sdk(self, instance: dict, max_step: int, timelimit: int) -> list[dict]:
        self.container.send_command(
            f"""
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed. Installing Python 3.12..."
    sudo apt-get update && sudo apt-get install -y python3.12
else
    echo "Python is already installed."
fi
"""
        )
        self.container.send_command("python3 -m pip install claude-code-sdk")
        with open("src/agent/cc/claude_code_main.py.template") as f:
            entrance_template = f.read()
        entrance_template.replace("SYS_PROMPT", self.system_prompt).replace(
            "PROMPT", self.user_prompt.format(description=instance["problem_statement"].replace('"""', "'''"))
        ).replace("MAX_STEP", str(max_step))
        self.container.send_command(f"cat > /tmp/claude_code_main.py <<'CC_MAIN'\n{entrance_template}\nCC_MAIN\n")
        self.container.send_command("python3 /tmp/claude_code_main.py", timelimit * 60)
        return

    def run_instance(
        self, instance: dict, max_step: int = 40, timelimit: int = 120, run_method: Literal["python", "cli"] = "python"
    ) -> AgentResult:
        """
        timelimit: in minute
        """
        if run_method == "python":
            raise NotImplementedError("Claude Code Python SDK has not been fully implemented...")
            # traj = self._run_python_sdk(instance, max_step, timelimit)
        elif run_method == "cli":
            traj = self._run_cli(instance, max_step, timelimit)
        else:
            raise ValueError(f"wrong run_method {run_method}, run_method should be in [python, cli]")
        solution_patch = self.container.send_command("git --no-pager diff HEAD --diff-filter=M --text").output
        solution_patch = solution_patch.replace("git --no-pager diff HEAD --diff-filter=M --text\n", "")
        reproduction_file = self.container.send_command("cat /testbed/reproduction.py").output
        reproduction_file = reproduction_file.replace("cat /testbed/reproduction.py\n", "")
        return_value: AgentResult = {
            "instance_id": instance["instance_id"],
            "model_patch": solution_patch,
            "reproduction_file": reproduction_file,
            "model_name_or_path": "cc",
            "trajectory": traj,
            "success": 0.0,
        }
        return return_value

    def calculate_intermediate_rewards_per_slice(
        self, gold_patch: str, model_patch: str, reproduction_file: str, trajectory: ClaudeCodeTraj
    ) -> list[tuple[ClaudeCodeStep, float]]:
        reward_estimator: RewardEstimatorWholeSlice = RewardEstimatorWholeSlice(
            self.container, reproduction_file, model_patch, gold_patch
        )
        return reward_estimator.main(trajectory)

    def __del__(self):
        if hasattr(self, "container"):
            self.container.cleanup()
