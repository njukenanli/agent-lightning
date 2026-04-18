import datetime
import os


def logger(run_id: str, instance_id: str, sample_id: str, text: str):
    os.makedirs(f"./logs/{run_id}/{instance_id}", exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"./logs/{run_id}/{instance_id}/{sample_id}.txt", mode="a") as f:
        print(f"\n\n{current_time}\n{text}\n", file=f)
