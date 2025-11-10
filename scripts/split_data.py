import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV
csv_path = "./data/triplets.csv"
df = pd.read_csv(csv_path)

# Extract subjects from paths (assumes folder names have ADNI_<site>_S_<subject>)
df["subject"] = df["first_path"].apply(lambda p: p.split("/")[-2])  # folder name

# Get unique subjects
subjects = df["subject"].unique()

# Split subjects into train / eval
train_subjects, eval_subjects = train_test_split(subjects, test_size=0.2, random_state=42)

# Filter triplets
train_df = df[df["subject"].isin(train_subjects)].reset_index(drop=True)
eval_df  = df[df["subject"].isin(eval_subjects)].reset_index(drop=True)

# Optional: save split CSVs
train_df.to_csv("./data/triplets_train.csv", index=False)
eval_df.to_csv("./data/triplets_eval.csv", index=False)