import datetime
import os


def logger(run_id, instance_id, text):
    os.makedirs(f"./logs/{run_id}", exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"./logs/{run_id}/{instance_id}", mode="a") as f:
        print(f"\n\n{current_time}\n{text}\n", file=f)
