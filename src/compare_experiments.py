import mlflow

runs = mlflow.search_runs()

best_run = runs.sort_values(
    by="metrics.f1",
    ascending=False
).iloc[0]

print("Best Run")
print(best_run)