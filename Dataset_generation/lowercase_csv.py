import pandas as pd

df = pd.read_csv("clickbait_generated_train_val.csv")

df = df.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))

df.to_csv("clickbait_generated_train_val.csv", index=False)
