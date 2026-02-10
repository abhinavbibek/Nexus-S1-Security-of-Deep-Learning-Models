import pandas as pd

df = pd.read_csv("results/all_metrics.csv")

def table1(dataset):
    sub = df[df.dataset == dataset]
    rows = []
    for eps in [0.05, 0.1, 0.2]:
        eps_df = sub[sub.epsilon == eps]
        success = (eps_df.tmr <= 0.01).sum()
        total = len(eps_df)
        rows.append({
            "epsilon": eps,
            "ISPL+B": f"{success} / {total}"
        })
    return pd.DataFrame(rows)

table1("cifar10").to_csv("results/table1_cifar10.csv", index=False)
table1("gtsrb").to_csv("results/table1_gtsrb.csv", index=False)
