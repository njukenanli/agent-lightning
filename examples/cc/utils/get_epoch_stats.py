import os, json

from examples.cc.utils.reward import RewardEstimatorWholeSlice

base_dir = "test_res"

stats = {
    "success_instance_count": 0,
    "failure_instance_count": 0,
    "effective_edit_for_failure_count": 0,
    "effective_edit_for_failure_reward": 0.0,
    "success_localization_reward": 0.0,
    "failure_localization_reward": 0.0,
    "success_reproduction_reward": 0.0,
    "failure_reproduction_reward": 0.0,
    "success_edit_reward": 0.0,
    "failure_edit_reward": 0.0,
    "success_submit_reward": 0.0,
    "failure_submit_reward": 0.0,
}

for file in os.listdir(base_dir):
    with open(f"{base_dir}/{file}") as f:
        res: RewardEstimatorWholeSlice.ReturnType = json.load(f)
    if res["overall"]["success"] > 0.5:
        stats["success_instance_count"] += 1
        stats["success_localization_reward"] += res["overall"]["localization"]
        stats["success_reproduction_reward"] += res["overall"]["repdocution"]
        stats["success_edit_reward"] += res["overall"]["edit"]
        stats["success_submit_reward"] += res["overall"]["result"]
    else:
        stats["failure_instance_count"] += 1
        stats["failure_localization_reward"] += res["overall"]["localization"]
        stats["failure_reproduction_reward"] += res["overall"]["repdocution"]
        stats["failure_edit_reward"] += res["overall"]["edit"]
        stats["failure_submit_reward"] += res["overall"]["result"]
    for slice in res["slices"]:
        if slice["traj"]["process"] == "Edit":
            if slice["reward"] > -0.99:
                stats["effective_edit_for_failure_count"] += 1
                stats["effective_edit_for_failure_reward"] += slice["reward"]

print("success_avg ; failure_avg ; overall_avg")
print("Localization", 
      stats["success_localization_reward"]/stats["success_instance_count"],
      stats["failure_localization_reward"]/stats["failure_instance_count"],
      (stats["success_localization_reward"]+stats["failure_localization_reward"])/(stats["success_instance_count"]+stats["failure_instance_count"]),
      sep = " ; ")
print("Reproduction", 
      stats["success_reproduction_reward"]/stats["success_instance_count"],
      stats["failure_reproduction_reward"]/stats["failure_instance_count"],
      (stats["success_reproduction_reward"]+stats["failure_reproduction_reward"])/(stats["success_instance_count"]+stats["failure_instance_count"]),
      sep = " ; ")
print("Edit", 
      stats["success_edit_reward"]/stats["success_instance_count"],
      stats["failure_edit_reward"]/stats["failure_instance_count"],
      (stats["success_edit_reward"]+stats["failure_edit_reward"])/(stats["success_instance_count"]+stats["failure_instance_count"]),
      sep = " ; ")
print("Submit", 
      stats["success_submit_reward"]/stats["success_instance_count"],
      stats["failure_submit_reward"]/stats["failure_instance_count"],
      (stats["success_submit_reward"]+stats["failure_submit_reward"])/(stats["success_instance_count"]+stats["failure_instance_count"]),
      sep = " ; ")
print("Effective edit evg in failed trajs", 
      stats["effective_edit_for_failure_reward"],
      stats["effective_edit_for_failure_count"],
      stats["effective_edit_for_failure_reward"]/stats["effective_edit_for_failure_count"])