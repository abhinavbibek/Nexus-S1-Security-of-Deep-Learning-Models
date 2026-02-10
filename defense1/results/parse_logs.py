import re
import glob
import pandas as pd


def parse_log(path):
    with open(path, "r") as f:
        txt = f.read()

    # -------------------------
    # Dataset line
    # -------------------------
    m_ds = re.search(r"Dataset:\s+(.*)", txt)
    if m_ds is None:
        raise ValueError("Dataset line not found")

    ds = m_ds.group(1)

    # -------------------------
    # Source / Target / Epsilon
    # Robust regex: capture only valid float
    # -------------------------
    m = re.search(r"backdoor-(\d+)-to-(\d+)-(0\.\d+)", ds)
    if m is None:
        raise ValueError("Could not parse source/target/epsilon")

    source = int(m.group(1))
    target = int(m.group(2))
    eps = float(m.group(3))   # SAFE: no trailing dot possible

    # -------------------------
    # Metrics
    # -------------------------
    clean_acc = float(
        re.search(r"clean accuracy:\s*([\d.]+)", txt).group(1)
    )

    tmr = float(
        re.search(r"targeted misclassification:\s*([\d.]+)", txt).group(1)
    )

    pmr = float(
        re.search(r"poison misclassification:\s*([\d.]+)", txt).group(1)
    )

    return {
        "dataset": "cifar10" if "cifar" in ds.lower() else "gtsrb",
        "source": source,
        "target": target,
        "epsilon": eps,
        "clean_acc": clean_acc,
        "tmr": tmr,
        "pmr": pmr,
        "logfile": path,
    }


# -------------------------
# Main
# -------------------------
rows = []

paths = glob.glob("../logs/*/run_*.txt")
print("Found log files:", len(paths))

for p in paths[:5]:
    print("  ", p)

for path in paths:
    try:
        rows.append(parse_log(path))
    except Exception as e:
        print("FAILED:", path, "|", e)

df = pd.DataFrame(rows)

print("\nParsed rows:", len(df))
print(df.head())

df.to_csv("all_metrics.csv", index=False)
print("\nSaved -> all_metrics.csv")
