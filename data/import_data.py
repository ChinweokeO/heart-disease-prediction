from ucimlrepo import fetch_ucirepo

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = X.copy()
df["target"] = y

df.to_csv("data/heart_disease.csv", index=False)