import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def plot_test_split_distribution(
    csv_path,
    save_path="agedb_test_age_distribution.png",
    max_age=100
):
    # -----------------------------
    # 1. Load CSV
    # -----------------------------
    df = pd.read_csv(csv_path)

    assert {"age", "split"}.issubset(df.columns), \
        "CSV must contain columns: age, split"

    # -----------------------------
    # 2. Filter test split
    # -----------------------------
    df_test = df[df["split"] == "test"].copy()
    ages = df_test["age"].astype(int)

    # 可选：限制最大年龄
    ages = ages[ages <= max_age]

    # -----------------------------
    # 3. Count bins (bin = 1)
    # -----------------------------
    counts = Counter(ages)
    bins = np.arange(0, max_age + 1)
    values = np.array([counts.get(b, 0) for b in bins])

    print(f"Total test samples: {values.sum()}")

    # -----------------------------
    # 4. Plot
    # -----------------------------
    plt.figure(figsize=(14, 5))

    plt.bar(
        bins,
        values,
        color="#3b82f6",
        edgecolor="black",
        linewidth=0.3
    )

    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)
    plt.title("AgeDB Test Split Age Distribution (bin = 1)", fontsize=16)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.margins(x=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_test_split_distribution(
        csv_path="/home/ydubf/imbalanced-regression/agedb-dir/data/agedb.csv",   # 👈 改成你的路径
        save_path="/home/ydubf/imbalanced-regression/agedb-dir/data/agedb_test_bin_dist.png",
        max_age=100
    )