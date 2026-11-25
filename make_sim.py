# make_sim.py: run in the hour-detective folder
import os, shutil
import pandas as pd

os.makedirs("sim", exist_ok=True)
df = pd.read_csv("train.csv").sample(frac=1, random_state=7).reset_index(drop=True)
label_cols = ["hours_spent", "log_hours", "hours_bucket"]

df.iloc[:4000].to_csv("sim/train.csv", index=False)
df.iloc[4000:].drop(columns=label_cols).to_csv("sim/test.csv", index=False)
df.iloc[4000:][["project_id", "hours_spent"]].to_csv("sim/answers.csv", index=False)
shutil.copy("predict.py", "sim/predict.py")
print("sim folder ready")