from __future__ import annotations

import base64
from difflib import SequenceMatcher
import json
import os
from typing import Any, TypedDict
import uuid

import unidiff
from src.utils.docker_runtime import CommandResult, Runtime
from src.utils.type import AgentResult, ClaudeCodeStep, ClaudeCodeTraj, TrajSlice


class TajectoryProcessor:
    mapping = {
        "Read": "Localization",
        "Glob": "Localization",
        "Grep": "Localization",
        "Write": "Reproduction",
        "Edit": "Edit",
        "BashAfterEdit": "Validation",
        "Summary|Result": "Result",
    }

    @staticmethod
    def split(traj: ClaudeCodeTraj, slice_each_edit: bool = True) -> list[TrajSlice]:
        slices: list[TrajSlice] = []
        cur_act = "Localization"
        next_act = ""
        start = -1
        end = -1
        for idx in range(len(traj)):
            if traj[idx]["type"] == "result":
                if traj[idx - 1]["type"] == "assistant" and traj[idx - 1]["message"]["content"][0]["type"] == "text":
                    slices.append(
                        {"process": cur_act, "step_range": (start, idx - 2), "content": traj[start : idx - 1]}
                    )
                    slices.append(
                        {"process": "Result", "step_range": (idx - 1, idx), "content": traj[idx - 1 : idx + 1]}
                    )
                else:
                    slices.append({"process": cur_act, "step_range": (start, idx - 1), "content": traj[start:idx]})
                    slices.append({"process": "Result", "step_range": (idx, idx), "content": traj[idx : idx + 1]})
                break
            if traj[idx]["type"] == "system":
                continue
            else:
                if start == -1:
                    start = idx
            if traj[idx]["type"] == "user":
                continue
            if traj[idx]["type"] == "assistant":
                if traj[idx]["parent_tool_use_id"] is not None:
                    # is subtask of sugagent
                    continue
                assert len(traj[idx]["message"]["content"]) == 1, traj[idx]["message"]["content"]
                if traj[idx]["message"]["content"][0]["type"] == "text":
                    continue
                content = traj[idx]["message"]["content"][0]
                if content["type"] == "tool_use":
                    # List events
                    if cur_act == "Edit" and content["name"] == "Bash":
                        next_act = "Validation"
                    elif cur_act == "Validation" and content["name"] == "Bash":
                        continue
                    elif content["name"] == "Write" and cur_act != "Reproduction":
                        next_act = "Reproduction"
                    elif content["name"] in {"Read", "Write", "Glob"} and cur_act != "Localization":
                        next_act = "Localization"
                    elif content["name"] == "Edit" and cur_act != "Edit":
                        next_act = "Edit"
                    elif slice_each_edit and content["name"] == "Edit" and cur_act == "Edit":
                        # We take each edit action as one slice in this setting
                        next_act = "Edit"
                    else:
                        # If no event to trigger, do not update slices list
                        continue

                    for end in range(idx, 0, -1):
                        if traj[end]["type"] != "assistant":
                            break

                    if (
                        len(slices) >= 2
                        and cur_act == "Reproduction"
                        and slices[-1]["process"] == "Localization"
                        and slices[-2]["process"] == "Reproduction"
                    ):
                        # merge
                        slices[-2]["step_range"] = (slices[-2]["step_range"][0], end)
                        slices[-2]["content"] = traj[slices[-2]["step_range"][0] : end + 1]
                        slices.pop()  # default last

                    else:
                        slices.append(
                            {"process": cur_act, "step_range": (start, end), "content": traj[start : end + 1]}
                        )
                    start = end + 1
                    cur_act = next_act
        return slices

    @staticmethod
    def extract_err(traj: ClaudeCodeTraj) -> list[TrajSlice]:
        err_list: list[TrajSlice] = []
        for idx in range(len(traj)):
            if traj[idx]["type"] == "user" and "<tool_use_error>" in traj[idx]["message"]["content"][0]["content"]:
                assert len(traj[idx]["message"]["content"]) == 1, traj[idx]["message"]["content"]
                begin = None
                for begin in range(idx - 1, 0, -1):
                    if traj[begin]["type"] == "user":
                        break
                if begin is not None:
                    err_list.append(
                        {
                            "process": "InvalidToolCall",  # In [ToolFormatError, ToolNameError, StringReplaceNoChange, StringReplaceNotFound]
                            "step_range": (begin + 1, idx),
                            "content": traj[begin + 1 : idx + 1],
                        }
                    )
        return err_list


class BaseRewardEstimator:
    reward_range = {
        "Localization": "{ -1 } | (0, 1]",
        "Reproduction": "{ -1, 1 }",
        "Edit": "[-1, 1]",
        "Validation": "{ 0.5 }",
        "Result": "{ -1, -0.5, 0.5 }",
        "InvalidToolCall": "{ -1 }",
    }

    class ParsedPatch(TypedDict):
        start: int
        end: int
        deleted_lines: list[str]
        added_lines: list[str]

    class SliceReward(TypedDict):
        traj: TrajSlice
        keysteps: ClaudeCodeTraj
        reward: float  # [-1.0 , 1.0] in practice

    class ReturnType(TypedDict):
        steps: list[tuple[ClaudeCodeStep, str, float]] # step info ; type in ["Localization", "Edit", "Reproduction", "Validation", "Result"] ; reward
        slices: list[BaseRewardEstimator.SliceReward]
        overall: dict[str, float]

    def __init__(self, container: Runtime, reproduction_file: str, solution_patch: str, gold_patch: str):

        self.container: Runtime = container
        self.reproduction_file: str = reproduction_file
        self.solution_patch: str = solution_patch
        self.gold_patch: str = gold_patch
        self.last_sim: float = 0.0
        self.last_diff: str = ""
        self.parsed_gold_patch: dict[str, list[BaseRewardEstimator.ParsedPatch]] = self._parse_patch(
            self.gold_patch
        )
        self.total_lines_gold_patch: int = 0 # the target / added block lineno
        self.sorted_added_lines: str = ""
        for file_path in sorted(self.parsed_gold_patch.keys()):
            for loc in sorted(self.parsed_gold_patch[file_path], key=lambda x: x["start"]):
                self.total_lines_gold_patch += loc["end"] - loc["start"]
                self.sorted_added_lines += "\n".join(loc["added_lines"]) + "\n"
        self.container.send_command("cd /testbed; git stash; git reset --hard HEAD;")

    @property
    def current_patch(self) -> str:
        return self.container.send_command("git --no-pager diff HEAD --diff-filter=M --text").output.replace(
            "git --no-pager diff HEAD --diff-filter=M --text\n", ""
        )

    def read_file(self, file_path: str) -> str:
        return self.container.send_command(f"cat {file_path}").output.replace(f"cat {file_path}\n", "")
    
    def count_lines(self, file_path: str) -> int:
        num = (self.container.send_command(f"wc -l {file_path}")
               .output.replace(f"wc -l {file_path}\n", "")
               .split()[0])
        try:
            num = int(num)
        except:
            num = 0
        return num

    def write_file(self, file_path: str, content: str) -> bool:
        return (
            self.container.send_command(
                f"cat > {file_path} <<'REPRODUCTION_FILE'\n{content}\nREPRODUCTION_FILE\n"
            ).metadata.exit_code
            == 0
        )

    def apply_patch(self, patch: str) -> bool:
        return self.container.send_command(f"""git apply - <<'NEW_PATCH'\n{patch}\nNEW_PATCH""").metadata.exit_code == 0

    def string_replace(self, file_path: str, old_string: str, new_string: str) -> bool:
        old_file_name = f"{uuid.uuid4()}.txt"
        new_file_name = f"{uuid.uuid4()}.txt"
        script_name = f"{uuid.uuid4()}.py"
        script = f"""
with open("/testbed/mnt_tmp/{old_file_name}", encoding="utf-8") as f:
    old_string = f.read()
with open("/testbed/mnt_tmp/{new_file_name}", encoding="utf-8") as f:
    new_string = f.read()
with open("{file_path}", encoding="utf-8") as f:
    file_content = f.read()
if old_string not in file_content:
    raise ValueError(f"Old string not found!")
file_content = file_content.replace(old_string, new_string)
with open("{file_path}", "w", encoding="utf-8") as f:
    f.write(file_content)
"""
        with open(os.path.join(os.getcwd(), "logs", "tmp", old_file_name), "w") as f:
            f.write(old_string)
        with open(os.path.join(os.getcwd(), "logs", "tmp", new_file_name), "w") as f:
            f.write(new_string)
        with open(os.path.join(os.getcwd(), "logs", "tmp", script_name), "w") as f:
            f.write(script)
        res = self.container.send_command(f"python /testbed/mnt_tmp/{script_name}")
        self.container.send_command(f"rm -f /testbed/mnt_tmp/{old_file_name}")
        self.container.send_command(f"rm -f /testbed/mnt_tmp/{new_file_name}")
        self.container.send_command(f"rm -f /testbed/mnt_tmp/{script_name}")
        if res.metadata.exit_code == 0:
            return True
        else:
            #print(res.output)
            return False

    @staticmethod
    def _parse_patch(patch: str) -> dict[str, list[ParsedPatch]]:
        '''
        use unidiff to get file_path : [(start1, end1, deleted_lines1, added_lines1), (start2, end2, deleted_lines2, added_lines2) ...] mapping
        The added and deleted lines should not have + - and '\n'
        The start and end lineno is the target (added) one.
        '''

        result: dict[str, list[BaseRewardEstimator.ParsedPatch]] = {}

        try:
            patch_set = unidiff.PatchSet(patch)
        except Exception:
            # If patch parsing fails, return empty dict
            return result

        for patched_file in patch_set:
            file_path = patched_file.path
            patches: list[BaseRewardEstimator.ParsedPatch] = []

            for hunk in patched_file:
                deleted_lines: list[str] = []
                added_lines: list[str] = []
                start = hunk.target_start
                end = hunk.target_start + hunk.target_length - 1

                for line in hunk:
                    if line.is_removed:
                        # Remove the leading '-' and trailing newline
                        deleted_lines.append(line.value.rstrip("\n"))
                    elif line.is_added:
                        # Remove the leading '+' and trailing newline
                        added_lines.append(line.value.rstrip("\n"))

                patches.append({"start": start, "end": end, "deleted_lines": deleted_lines, "added_lines": added_lines})

            if patches:
                result[file_path] = patches

        return result

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        res: list[tuple[int, int]] = []
        intervals = sorted(intervals)
        cur = list(intervals[0])
        for interval in intervals[1:]:
            if interval[0] <= cur[1]:
                if interval[1] > cur[1]:
                    cur[1] = interval[1]
            elif interval[0] - 1 == cur[1]:
                cur[1] = interval[1]
            else:
                res.append(tuple(cur))
                cur = list(interval)
        res.append(tuple(cur))
        return res
    
    @staticmethod
    def sort_flattern(slices: list[BaseRewardEstimator.SliceReward]) -> list[tuple[ClaudeCodeStep, str, float]]:
        '''
        Note: this means sort by step idx first and then flattern
        '''
        slices = sorted(slices, key=lambda x: x["traj"]["step_range"][0])
        res: list[tuple[ClaudeCodeStep, str, float]] = []
        for slice in slices:
            for step in slice["traj"]["content"]:
                res.append((step, slice["traj"]["process"], slice["reward"]))
        return res

    @staticmethod
    def filter_non_assistant_msg(steps: list[tuple[ClaudeCodeStep, str, float]]) -> list[tuple[ClaudeCodeStep, str, float]]:
        res: list[tuple[ClaudeCodeStep, str, float]] = []
        for step in steps:
            if step[0]["type"] != "assistant":
                continue
            res.append(step)
        return res
    
    @staticmethod
    def merge_text_toolcall_msg(steps: list[tuple[ClaudeCodeStep, str, float]]) -> list[tuple[ClaudeCodeStep, str, float]]:
        res: list[tuple[ClaudeCodeStep, str, float]] = []
        cached_step: tuple[ClaudeCodeStep, str, float] | None = None
        for step in steps:
            if step[0]["type"] != "assistant":
                if cached_step is not None:
                    res.append(cached_step)
                    cached_step = None
                res.append(step)
            else:
                if cached_step is None:
                    cached_step = step
                else:
                    cached_step[0]["message"]["content"].extend(step[0]["message"]["content"])
        if cached_step is not None:
            res.append(cached_step)
        return res

    @staticmethod
    def sort_step(traj: ClaudeCodeTraj, steps: list[tuple[ClaudeCodeStep, str, float]]) -> list[tuple[ClaudeCodeStep, str, float]]:
        order: list[str] = []
        for step in traj:
            if step["message"] is not None and step["message"]["id"] not in order:
                order.append(step["message"]["id"])
        res: list[tuple[ClaudeCodeStep, str, float]] = []
        indices: dict[str, list[tuple[ClaudeCodeStep, str, float]]] = {}
        for i in steps:
            if i[0]["message"] is not None:
                if i[0]["message"]["id"] not in indices.keys():
                    indices[i[0]["message"]["id"]] = [i]
                else:
                    indices[i[0]["message"]["id"]].append(i)
        for msg_id in order:
            if msg_id in indices.keys():
                res.extend(indices[msg_id])
        return res
    
    @staticmethod
    def save_res(epoch: int, instance_id: str, processed_res: BaseRewardEstimator.ReturnType):
        os.makedirs(f"logs/result/epoch_{epoch}", exist_ok=True)
        with open(f"logs/result/epoch_{epoch}/{instance_id}.json", "w") as f:
            json.dump(processed_res,f,indent=True)


class RewardEstimatorWholeSlice(BaseRewardEstimator):
    """
    This RewardEstimator assigns all the steps in a whole Slice the same reward.
    """

    def localization(self, slice: TrajSlice) -> BaseRewardEstimator.SliceReward:
        """
        Reward: Recall of ground truth modified lines
        """
        effective_lines: int = 0
        keysteps: ClaudeCodeTraj = []
        viewd_locs: dict[str, list[tuple[int, int]]] = {}
        for step_id in range(len(slice["content"])):
            step = slice["content"][step_id]
            if step.get("message", None) is None:
                continue
            if (
                step["message"]["content"][0]["type"] == "tool_use"
                and step["message"]["content"][0]["name"].lower() == "read"
                and step["message"]["content"][0]["input"].get("file_path", None) is not None
            ):
                args: dict[str, str | int] = step["message"]["content"][0]["input"]
                file_path: str = args["file_path"].replace("/testbed/", "")
                if file_path not in self.parsed_gold_patch.keys():
                    continue
                target_locs: list[BaseRewardEstimator.ParsedPatch] = self.parsed_gold_patch[file_path]
                if args.get("offset", None) is not None and args.get("limit", None) is not None:
                    interval: tuple[int, int] = (args["offset"], args["offset"] + args["limit"])
                    if file_path in viewd_locs.keys():
                        viewd_locs[file_path].append(interval)
                    else:
                        viewd_locs[file_path] = [interval]
                    for target_loc in target_locs:
                        if (
                            target_loc["start"] <= interval[1] <= target_loc["end"]
                            or target_loc["start"] <= interval[0] <= target_loc["end"]
                        ):
                            if slice["content"][step_id - 1]["type"] == "assistant":
                                keysteps.append(slice["content"][step_id - 1])
                            keysteps.append(step)
                            break
                else:
                    if slice["content"][step_id - 1]["type"] == "assistant":
                        keysteps.append(slice["content"][step_id - 1])
                    keysteps.append(step)
                    interval = (0, self.count_lines(file_path) - 1)
                    viewd_locs[file_path] = [interval]
        for file_path in viewd_locs.keys():
            viewd_locs[file_path] = self._merge_intervals(viewd_locs[file_path])
            for target_loc in self.parsed_gold_patch[file_path]:
                for pred_loc in viewd_locs[file_path]:
                    effective_interval = (
                        max(target_loc["start"], pred_loc[0]),
                        min(target_loc["end"], pred_loc[1]),
                    )
                    effective_lines += max(0, effective_interval[1] - effective_interval[0])

        return {
            "keysteps": keysteps,
            "reward": -1.0 if effective_lines == 0 else min(1.0, effective_lines / self.total_lines_gold_patch),
            "traj": slice,
        }

    def reproduction(self, slices: list[TrajSlice]) -> list[BaseRewardEstimator.SliceReward]:
        """
        reward: whether reproduction.py exits with code from non zero to zero when gold_patch applied
        """
        self.container.send_command("git stash; git reset --hard HEAD;")
        keysteps: ClaudeCodeTraj = []
        merged_slice: ClaudeCodeTraj = []
        for slice in slices:
            merged_slice.extend(slice["content"])
        for step_id in range(len(merged_slice) - 1, 0, -1):
            if merged_slice[step_id].get("message", None) is None:
                continue
            if (
                merged_slice[step_id]["message"]["content"][0]["type"] == "tool_use"
                and merged_slice[step_id]["message"]["content"][0]["name"].lower() == "write"
            ):
                if merged_slice[step_id - 1]["type"] == "assistant":
                    keysteps.append(merged_slice[step_id - 1])
                keysteps.append(merged_slice[step_id])

        command = "python /testbed/reproduction.py"
        self.write_file("/testbed/reproduction.py", self.reproduction_file)
        pre_patch_res: CommandResult = self.container.send_command(command)
        if int(pre_patch_res.metadata.exit_code) == 0:
            return [
                {
                    "keysteps": keysteps,
                    "reward": -1.0,
                    "traj": slice,
                }
                for slice in slices
            ]
        self.apply_patch(self.gold_patch)
        post_patch_res: CommandResult = self.container.send_command(command)
        if int(post_patch_res.metadata.exit_code) != 0:
            return [
                {
                    "keysteps": keysteps,
                    "reward": -1.0,
                    "traj": slice,
                }
                for slice in slices
            ]
        return [
            {
                "keysteps": keysteps,
                "reward": 1.0,
                "traj": slice,
            }
            for slice in slices
        ]

    def _get_changed_lines(
        self, old_patch_parsed: dict[str, list[ParsedPatch]], new_patch_parsed: dict[str, list[ParsedPatch]]
    ) -> dict[str, list[tuple[int, int]]]:
        res: dict[str, list[tuple[int, int]]] = {}
        for new_path in new_patch_parsed.keys():
            if new_path not in old_patch_parsed.keys():
                res[new_path] = [(loc["start"], loc["end"]) for loc in new_patch_parsed[new_path]]
            else:
                res[new_path] = []
                old_loc_index: dict[tuple[int, int], str] = {}
                for old_loc in old_patch_parsed[new_path]:
                    old_loc_index[(old_loc["start"], old_loc["end"])] = ("\n".join(old_loc["added_lines"])).strip()
                for new_loc in new_patch_parsed[new_path]:
                    cur_interval = (new_loc["start"], new_loc["end"])
                    if cur_interval not in old_loc_index.keys():
                        res[new_path].append(cur_interval)
                    else:
                        if ("\n".join(new_loc["added_lines"])).strip() != old_loc_index[cur_interval]:
                            res[new_path].append(cur_interval)
        return res

    @staticmethod
    def code_similarity(a: str, b: str) -> float:
        """Ratcliff-Obershelp similarity algorithm"""
        return SequenceMatcher(None, a, b).ratio()

    def edit(self, slice: TrajSlice) -> BaseRewardEstimator.SliceReward:
        """
        reward = old similarity - new similarity of added lines compared to ground truth added lines
        """
        self.container.send_command("git stash; git reset --hard HEAD;")
        old_added_lines: str = ""
        old_modification: str = self.current_patch
        old_parsed_modification = self._parse_patch(old_modification)
        for file_patch in sorted(old_parsed_modification.keys()):
            for loc in sorted(old_parsed_modification[file_patch], key=lambda x: x["start"]):
                old_added_lines += "\n".join(loc["added_lines"]) + "\n"
        keysteps: ClaudeCodeTraj = []
        for step_id in range(len(slice["content"])):
            step = slice["content"][step_id]
            if step.get("message", None) is None:
                continue
            if (
                step["message"]["content"][0]["type"] == "tool_use"
                and step["message"]["content"][0]["name"].lower() == "edit"
                and (not slice["content"][step_id + 1]["message"]["content"][0].get("is_error", False))
            ):
                if slice["content"][step_id - 1]["type"] == "assistant":
                    keysteps.append(slice["content"][step_id - 1])
                keysteps.append(step)
                self.string_replace(
                    step["message"]["content"][0]["input"]["file_path"],
                    step["message"]["content"][0]["input"]["old_string"],
                    step["message"]["content"][0]["input"]["new_string"],
                )
        new_modification: str = self.current_patch
        new_added_lines: str = ""
        new_parsed_modification = self._parse_patch(new_modification)
        modified_locs: dict[str, list[tuple[int, int]]] = self._get_changed_lines(
            old_parsed_modification, new_parsed_modification
        )
        # If no overlap, return -1.0
        has_overlap_with_gt = False
        for file_path in modified_locs.keys():
            for pred_locs in modified_locs[file_path]:
                if file_path not in self.parsed_gold_patch.keys():
                    continue
                else:
                    for gt_locs in self.parsed_gold_patch[file_path]:
                        s = max(gt_locs["start"], pred_locs[0])
                        e = min(gt_locs["end"], pred_locs[1])
                        if s <= e:
                            has_overlap_with_gt = True
                            break
                if has_overlap_with_gt:
                    break
            if has_overlap_with_gt:
                break
        if not has_overlap_with_gt:
            return {"keysteps": keysteps, "reward": -1.0, "traj": slice}
        for file_patch in sorted(new_parsed_modification.keys()):
            for loc in sorted(new_parsed_modification[file_patch], key=lambda x: x["start"]):
                new_added_lines += "\n".join(loc["added_lines"]) + "\n"
        # Ratcliff-Obershelp similarity algorithm
        old_sim: float = self.code_similarity(self.sorted_added_lines, old_added_lines)
        new_sim: float = self.code_similarity(self.sorted_added_lines, new_added_lines)
        reward = new_sim - old_sim
        print(reward)
        return {
            "keysteps": keysteps,
            "reward": reward,
            "traj": slice,
        }

    def successful_edit(self, slice: TrajSlice) -> BaseRewardEstimator.SliceReward:
        """
        reward = old similarity - new similarity of added lines compared to ground truth added lines
        """

        keysteps: ClaudeCodeTraj = []
        for step_id in range(len(slice["content"])):
            step = slice["content"][step_id]
            if step.get("message", None) is None:
                continue
            if (
                step["message"]["content"][0]["type"] == "tool_use"
                and step["message"]["content"][0]["name"].lower() == "edit"
                and (not slice["content"][step_id + 1]["message"]["content"][0].get("is_error", False))
            ):
                if slice["content"][step_id - 1]["type"] == "assistant":
                    keysteps.append(slice["content"][step_id - 1])
                keysteps.append(step)

        return {
            "keysteps": keysteps,
            "reward": 1.0,
            "traj": slice,
        }

    @staticmethod
    def validation(slices: list[TrajSlice]) -> list[BaseRewardEstimator.SliceReward]:
        return [{"keysteps": slice["content"], "reward": 0.5, "traj": slice} for slice in slices]

    def result(self, slice: TrajSlice) -> BaseRewardEstimator.SliceReward:
        reward: float = 0.5
        # These two cases are when tool call format is wrong
        # so that tool call are parsed as text
        # which makes the agent early exit
        if "<tool_call>" in slice["content"][-1].get("result", ""):
            reward = -1.0
        elif (
            len(slice["content"]) > 1
            and slice["content"][-2]["message"]["content"][0]["type"] == "text"
            and "<tool_call>" in slice["content"][-2]["message"]["content"][0]["text"]
        ):
            reward = -1.0
        elif not self.solution_patch.strip():
            reward = -1.0
        else:
            self.container.send_command("git stash; git reset --hard HEAD;")
            self.write_file("/testbed/reproduction.py", self.reproduction_file)
            self.apply_patch(self.solution_patch)
            suc = self.container.send_command("python /testbed/reproduction.py").metadata.exit_code == 0
            if not suc:
                reward = -0.5
        
        return {
            "keysteps": slice["content"],
            "reward": reward,
            "traj": slice,
        }

    @staticmethod
    def assign_error_penalties(sequence: list[tuple[ClaudeCodeStep, str, float]]) -> list[tuple[ClaudeCodeStep, str, float]]:
        for step_id in range(len(sequence)):
            if sequence[step_id][0]["type"] == "user" \
                and "<tool_use_error>" in sequence[step_id][0]["message"]["content"][0].get("content", ""):
                find_assistant = False
                for back_id in range(step_id, -1, -1):
                    if find_assistant and sequence[back_id][0]["type"] != "assistant":
                        break
                    if not find_assistant and sequence[back_id][0]["type"] == "assistant":
                        find_assistant = True
                    sequence[back_id] = (sequence[back_id][0], sequence[back_id][1], -1.0)
        return sequence

    def calculate_reward(self, traj: ClaudeCodeTraj, overwrite_edit_with_final_success: bool) -> dict[str, list[BaseRewardEstimator.SliceReward]]:
        slices: list[TrajSlice] = TajectoryProcessor.split(traj)
        loc_slices: list[TrajSlice] = []
        repro_slices: list[TrajSlice] = []
        edit_slices: list[TrajSlice] = []
        validation_slices: list[TrajSlice] = []
        result_slice: TrajSlice | None = None
        for slice in slices:
            if slice["process"] == "Localization":
                loc_slices.append(slice)
            if slice["process"] == "Reproduction":
                repro_slices.append(slice)
            if slice["process"] == "Edit":
                edit_slices.append(slice)
            if slice["process"] == "Validation":
                validation_slices.append(slice)
            if slice["process"] == "Result":
                result_slice = slice
        assert result_slice is not None

        localization_reward_list: list[BaseRewardEstimator.SliceReward] = [
            self.localization(slice) for slice in loc_slices
        ]
        reproduction_reward_list: list[BaseRewardEstimator.SliceReward] = self.reproduction(repro_slices)
        if overwrite_edit_with_final_success:
            edit_reward_list: list[BaseRewardEstimator.SliceReward] = [self.successful_edit(slice) for slice in edit_slices]
        else:
            edit_reward_list: list[BaseRewardEstimator.SliceReward] = [self.edit(slice) for slice in edit_slices]
        validation_reward_list: list[BaseRewardEstimator.SliceReward] = self.validation(validation_slices)
        result_reward_list: list[BaseRewardEstimator.SliceReward] = [self.result(result_slice)]
        return {
            "reproduction": reproduction_reward_list,
            "localization": localization_reward_list,
            "edit": edit_reward_list,
            "validation": validation_reward_list,
            "result": result_reward_list,
        }
    
    def calculate_overall_edit_sim(self):
        parsed_solution_patch: dict[str, list[BaseRewardEstimator.ParsedPatch]] = self._parse_patch(
            self.solution_patch
        )
        preditions_sorted_added_lines: str = ""
        for file_path in sorted(parsed_solution_patch.keys()):
            for loc in sorted(parsed_solution_patch[file_path], key=lambda x: x["start"]):
                preditions_sorted_added_lines += "\n".join(loc["added_lines"]) + "\n"
        sim: float = self.code_similarity(self.sorted_added_lines, preditions_sorted_added_lines)
        return sim
    
    def calculate_overall_localization_sim(self, slices: list[ClaudeCodeTraj]) -> float:
        effective_lines: int = 0
        viewd_locs: dict[str, list[tuple[int, int]]] = {}
        for traj in slices:
            for step in traj:
                if step["message"] is None:
                    continue
                if (
                    step["message"]["content"][0]["type"] == "tool_use"
                    and step["message"]["content"][0]["name"].lower() == "read"
                    and step["message"]["content"][0]["input"].get("file_path", None) is not None
                ):
                    args: dict[str, str] = step["message"]["content"][0]["input"]
                    file_path: str = args["file_path"].replace("/testbed/", "")
                    if file_path not in self.parsed_gold_patch.keys():
                        continue
                    if args.get("offset", None) is not None and args.get("limit", None) is not None:
                        interval: tuple[int, int] = (int(args["offset"]), int(args["offset"]) + int(args["limit"]))
                        if file_path in viewd_locs.keys():
                            viewd_locs[file_path].append(interval)
                        else:
                            viewd_locs[file_path] = [interval]
                    else:
                        interval = (0, self.count_lines(file_path) - 1)
                        viewd_locs[file_path] = [interval]
        for file_path in viewd_locs.keys():
            viewd_locs[file_path] = self._merge_intervals(viewd_locs[file_path])
            for target_loc in self.parsed_gold_patch[file_path]:
                for pred_loc in viewd_locs[file_path]:
                    effective_interval = (
                        max(target_loc["start"], pred_loc[0]),
                        min(target_loc["end"], pred_loc[1]),
                    )
                    effective_lines += max(0, effective_interval[1] - effective_interval[0])

        assert effective_lines <= self.total_lines_gold_patch, f"{effective_lines} vs {self.total_lines_gold_patch}"
        return effective_lines / self.total_lines_gold_patch
    
    def main(self, traj: ClaudeCodeTraj, success: float, overwrite_edit_reward_with_final: bool = True) -> BaseRewardEstimator.ReturnType:
        rewards: dict[str, list[BaseRewardEstimator.SliceReward]] = self.calculate_reward(traj, success > 0.5 and overwrite_edit_reward_with_final)
        overall_reward: dict[str, float] = {}
        overall_reward["success"] = success
        overall_reward["edit"] = self.calculate_overall_edit_sim()
        overall_reward["localization"] = self.calculate_overall_localization_sim([i["traj"]["content"] for i in rewards["localization"]])
        overall_reward["reproduction"] = rewards["reproduction"][0]["reward"] \
                                            if rewards["reproduction"] else -1.0
        overall_reward["result"] = rewards["result"][0]["reward"] \
                                            if rewards["result"] else -1.0
        slices: list[BaseRewardEstimator.SliceReward] = (
            rewards["localization"]
            + rewards["reproduction"]
            + rewards["edit"]
            + rewards["validation"]
            + rewards["result"]
        )

        reward_sequence = self.sort_flattern(slices)
        reward_sequence = self.merge_text_toolcall_msg(reward_sequence)
        reward_sequence = self.assign_error_penalties(reward_sequence)
        reward_sequence = self.filter_non_assistant_msg(reward_sequence)

        return {
            "steps": reward_sequence,
            "overall": overall_reward,
            "slices": slices,
        }
    
    @classmethod
    def intermediate_reward(cls,
                            result: AgentResult, 
                            container: Runtime, 
                            gold_patch: str,
                            epoch: int) -> BaseRewardEstimator.ReturnType:
        instance_id: str = result["instance_id"]
        reward_estimator = cls(container, 
                                result["reproduction_file"], 
                                result["model_patch"],
                                gold_patch)
        processed_res = reward_estimator.main(result["trajectory"], success=result["success"])
        reward_estimator.save_res(epoch, instance_id, processed_res)
        return processed_res

class RewardEstimatorWholeTraj(RewardEstimatorWholeSlice):
    """
    This RewardEstimator assigns all the steps in all slices with the same SliceType in a traj the same reward.
    """

    def main(self, traj: ClaudeCodeTraj, success: float, overwrite_edit_reward_with_final: bool = True) -> BaseRewardEstimator.ReturnType:
        slices: list[TrajSlice] = TajectoryProcessor.split(traj)
        loc_slices: list[TrajSlice] = []
        repro_slices: list[TrajSlice] = []
        edit_slices: list[TrajSlice] = []
        validation_slices: list[TrajSlice] = []
        result_slice: TrajSlice | None = None
        for slice in slices:
            if slice["process"] == "Localization":
                loc_slices.append(slice)
            if slice["process"] == "Reproduction":
                repro_slices.append(slice)
            if slice["process"] == "Edit":
                edit_slices.append(slice)
            if slice["process"] == "Validation":
                validation_slices.append(slice)
            if slice["process"] == "Result":
                result_slice = slice
        assert result_slice is not None

        overall_reward: dict[str, float] = {}
        overall_reward["success"] = success
        overall_reward["edit"] = self.calculate_overall_edit_sim()
        # edit reward: if the result is successful, the reward is 1.0 whatever the solution is. 
        # If failed, parts of the solution might still be correct or useful, so we still give the reward according to the edit similarity.
        edit_reward_list: list[BaseRewardEstimator.SliceReward] = [
            {
                "traj": edit_slice,
                "keysteps": [],
                "reward": 1.0 if (success > 0.5 and overwrite_edit_reward_with_final) else overall_reward["edit"]
            }
            for edit_slice in edit_slices
        ]
        overall_reward["localization"] = self.calculate_overall_localization_sim([i["content"] for i in loc_slices])
        loc_reward_list: list[BaseRewardEstimator.SliceReward] = [
            {
                "traj": loc_slice,
                "keysteps": [],
                "reward": overall_reward["localization"]
            }
            for loc_slice in loc_slices
        ]
        repro_reward_list = self.reproduction(repro_slices)
        overall_reward["reproduction"] = repro_reward_list[0]["reward"] \
                                            if repro_reward_list else -1.0
        validation_reward_list: list[BaseRewardEstimator.SliceReward] = self.validation(validation_slices)
        result_reward: BaseRewardEstimator.SliceReward = self.result(result_slice)
        overall_reward["result"] = result_reward["reward"] \
                                            if result_reward else -1.0


        reward_list: list[BaseRewardEstimator.SliceReward] = (
                        loc_reward_list 
                       + repro_reward_list 
                       + edit_reward_list 
                       + validation_reward_list
                       + [result_reward]
        )
        reward_sequence = self.sort_flattern(reward_list)
        reward_sequence = self.merge_text_toolcall_msg(reward_sequence)
        reward_sequence = self.assign_error_penalties(reward_sequence)
        reward_sequence = self.filter_non_assistant_msg(reward_sequence)

        return {
            "steps": reward_sequence,
            "overall": overall_reward,
            "slices": reward_list,
        }

    @classmethod
    def intermediate_reward(cls,
                            result: AgentResult, 
                            container: Runtime, 
                            gold_patch: str,
                            epoch: int) -> BaseRewardEstimator.ReturnType:
        instance_id: str = result["instance_id"]
        reward_estimator = cls(container, 
                                result["reproduction_file"], 
                                result["model_patch"],
                                gold_patch)
        processed_res = reward_estimator.main(result["trajectory"], success=result["success"])
        reward_estimator.save_res(epoch, instance_id, processed_res)
        return processed_res


def reward_test():
    from functools import partial
    import json, os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.utils.docker_runtime import Runtime
    from src.utils.logger import logger

    def proc_instance(sample: dict[str, Any], ds: dict[str, Any]) -> BaseRewardEstimator.ReturnType:
        instance_id = sample["instance_id"]
        image_name = f"swebench/sweb.eval.x86_64.{instance_id}".replace("__", "_1776_")
        if sample["success"] > 0.5:
            print("suceess", instance_id)
        container: Runtime = Runtime.start_session(image_name, 
                                                ds[instance_id], 
                                                partial(logger, run_id="reward_test", instance_id=instance_id),
                                                "linux")
        reward_estimator = RewardEstimatorWholeSlice(container, 
                                                sample["reproduction_file"], 
                                                sample["model_patch"],
                                                ds[instance_id]["patch"])
        processed_res = reward_estimator.main(sample["trajectory"], success=sample["success"])
        if 0.01 < processed_res["overall"]["localization"] < 0.99 and 0.01 < processed_res["overall"]["edit"] < 0.99:
            print("targeted", instance_id)
        return processed_res

    def process_and_save(sample: dict[str, Any], ds: dict[str, Any]) -> str | None:
        instance_id = sample["instance_id"]
        if os.path.exists(f"test_res/{instance_id}.json"):
            return None
        processed_res = proc_instance(sample, ds)
        with open(f"test_res/{instance_id}.json", "w") as f:
            json.dump(processed_res, f, indent=True)
        return instance_id

    #with open("utils/reward_test_sample_failed.jsonl") as f:
    #    samples = [json.loads(i) for i in f]
    samples: list[dict[str, Any]]=[]
    for file in os.listdir("result"):
        with open(f"result/{file}") as f:
            samples.append(json.load(f))
    with open("swebench_verified_filtered.jsonl") as f:
        ds = [json.loads(i) for i in f]
        ds = {i["instance_id"]: i for i in ds}

    os.makedirs("test_res", exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_and_save, sample, ds): sample["instance_id"] for sample in samples}
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {instance_id}: {e}")

if __name__ == "__main__":
    reward_test()

