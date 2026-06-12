import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def plot_boneage_distribution(
    csv_path,
    save_path="boneage_distribution.png",
    bin_size=1,        # bin=1 表示每个 boneage 一个 bin
    max_age=None       # 可选：限制最大 boneage
):
    # -----------------------------
    # 1. Load CSV
    # -----------------------------
    df = pd.read_csv(csv_path)

    # assert "boneage" in df.columns, "CSV must contain 'boneage' column"

    # 转成整数（保险）
    # boneage = df["boneage"].astype(int)
    boneage = df["Bone Age (months)"].astype(int)

    if max_age is not None:
        boneage = boneage[boneage <= max_age]

    # -----------------------------
    # 2. Count bins
    # -----------------------------
    if bin_size == 1:
        # 每个 age 一个 bin
        counts = Counter(boneage)
        ages = np.array(sorted(counts.keys()))
        values = np.array([counts[a] for a in ages])
    else:
        # 更粗的 bin（如 5 / 10）
        bins = (boneage // bin_size) * bin_size
        counts = Counter(bins)
        ages = np.array(sorted(counts.keys()))
        values = np.array([counts[a] for a in ages])

    # -----------------------------
    # 3. Print statistics
    # -----------------------------
    print("Boneage bin counts:")
    for a, v in zip(ages, values):
        print(f"  Age {a}: {v}")

    print(f"\nTotal samples: {values.sum()}")
    print(f"Min age: {boneage.min()}, Max age: {boneage.max()}")

    # -----------------------------
    # 4. Plot
    # -----------------------------
    plt.figure(figsize=(14, 5))

    plt.bar(
        ages,
        values,
        width=bin_size * 0.9,
        color="#3b82f6",
        edgecolor="black",
        linewidth=0.3
    )

    plt.xlabel("Bone Age", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)
    plt.title(f"BoneAge Distribution (bin={bin_size})", fontsize=16)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n[Saved] {save_path}")


# if __name__ == "__main__":
#     plot_boneage_distribution(
#         csv_path="/nfs/usrhome2/ydubf/BoneAge/train.csv",           # 👈 改成你的路径
#         save_path="/nfs/usrhome2/ydubf/BoneAge/boneage_train_dist.png",
#         bin_size=1,                     # 👈 每个 boneage 一个 bin
#         max_age=None                    # 或比如 240
#     )


# if __name__ == "__main__":
#     plot_boneage_distribution(
#         csv_path="/nfs/usrhome2/ydubf/BoneAge/val.csv",           # 👈 改成你的路径
#         save_path="/nfs/usrhome2/ydubf/BoneAge/boneage_val_dist.png",
#         bin_size=1,                     # 👈 每个 boneage 一个 bin
#         max_age=None                    # 或比如 240
#     )





import pandas as pd
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def make_balanced_boneage_txt(
    csv_path,
    output_txt,
    max_per_bin=200,
    seed=666,
    visualize=True,
    save_fig=True
):
    random.seed(seed)

    # -----------------------------
    # 1. load csv (BoneAge val.csv)
    # -----------------------------
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # -----------------------------
    # 2. build bins by rounded bone age (months)
    #    ⚠️ 仅用于分桶，不影响最终写入
    # -----------------------------
    bin_to_rows = defaultdict(list)

    for _, row in df.iterrows():
        bone_age = float(row["Bone Age (months)"])
        age_bin = int(round(bone_age))   # ⭐ 只用于 bin
        bin_to_rows[age_bin].append(row)

    # -----------------------------
    # 3. balanced sampling per bin
    # -----------------------------
    balanced_rows = []
    bin_counts = {}

    for age_bin in sorted(bin_to_rows.keys()):
        rows = bin_to_rows[age_bin]
        random.shuffle(rows)

        keep_n = min(len(rows), max_per_bin)  # 👈 样本少的 bin 全保留
        balanced_rows.extend(rows[:keep_n])
        bin_counts[age_bin] = keep_n

    print("Balanced bone-age bin counts:")
    for k, v in bin_counts.items():
        print(f"  Bin {k}: {v}")

    # -----------------------------
    # 4. save txt (⚠️ 连续 bone age!)
    # -----------------------------
    with open(output_txt, "w") as f:
        for row in balanced_rows:
            img_id = row["Image ID"]
            bone_age = row["Bone Age (months)"]
            f.write(f"{img_id} {bone_age}\n")

    print(f"Balanced BoneAge txt saved to: {output_txt}")
    print(f"Total balanced samples: {len(balanced_rows)}")

    # -----------------------------
    # 5. visualize distribution
    # -----------------------------
    if visualize:
        age_bins = [
            int(round(float(r["Bone Age (months)"])))
            for r in balanced_rows
        ]

        plt.figure(figsize=(8, 3))
        plt.hist(
            age_bins,
            bins=range(min(age_bins), max(age_bins) + 2),
            edgecolor="black"
        )
        plt.xlabel("Rounded Bone Age (months)")
        plt.ylabel("Count")
        plt.title("Balanced BoneAge Distribution")
        plt.tight_layout()

        if save_fig:
            fig_path = output_txt.replace(".txt", "_hist.png")
            plt.savefig(fig_path, dpi=200)
            print(f"Histogram saved to: {fig_path}")

        plt.show()


# -----------------------------
# main
# -----------------------------
# if __name__ == "__main__":
#     make_balanced_boneage_txt(
#         csv_path="/home/ydubf/imbalanced-regression/BoneAge/val.csv",
#         output_txt="/home/ydubf/imbalanced-regression/BoneAge/balanced_val_boneage.txt",
#         max_per_bin=30,   # 👈 和你 EF 代码一致
#         seed=666,
#         visualize=True,
#         save_fig=True
#     )



import pandas as pd

def merge_boneage_csv(
    train_csv,
    val_csv,
    output_csv
):
    # -----------------------------
    # 1. load csv
    # -----------------------------
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")

    # -----------------------------
    # 2. normalize column names
    # -----------------------------
    # train.csv: id,boneage,male
    train_df = train_df.rename(columns={
        "id": "id",
        "boneage": "boneage",
        "male": "male"
    })

    # val.csv: Image ID,male,Bone Age (months)
    val_df = val_df.rename(columns={
        "Image ID": "id",
        "Bone Age (months)": "boneage",
        "male": "male"
    })

    # -----------------------------
    # 3. keep only required columns
    # -----------------------------
    train_df = train_df[["id", "boneage", "male"]]
    val_df   = val_df[["id", "boneage", "male"]]

    # -----------------------------
    # 4. merge
    # -----------------------------
    merged_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Merged samples: {len(merged_df)}")

    # -----------------------------
    # 5. save
    # -----------------------------
    merged_df.to_csv(output_csv, index=False)
    print(f"[Saved] merged csv -> {output_csv}")

    # optional sanity check
    print("\nBoneage statistics:")
    print(merged_df["boneage"].describe())


# if __name__ == "__main__":
#     merge_boneage_csv(
#         train_csv="/home/ydubf/imbalanced-regression/BoneAge/train.csv",
#         val_csv="/home/ydubf/imbalanced-regression/BoneAge/val.csv",
#         output_csv="/home/ydubf/imbalanced-regression/BoneAge/merged_train_val.csv"
#     )





import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import random
import os

BASE_PATH = "/home/ydubf/imbalanced-regression/BoneAge"


def make_balanced_train_test(
    csv_name="merged_train_val.csv",
    max_size=30,
    seed=666,
    verbose=True,
    vis=True,
    save=True,
    save_fig=True
):
    random.seed(seed)

    file_path = join(BASE_PATH, csv_name)
    df = pd.read_csv(file_path)

    df["boneage"] = df["boneage"].astype(int)

    train_set = []
    test_set = []

    # -----------------------------
    # per-boneage bin split
    # -----------------------------
    for value in range(df["boneage"].min(), df["boneage"].max() + 1):
        curr_df = df[df["boneage"] == value]
        ids = curr_df["id"].values.tolist()

        if len(ids) == 0:
            continue

        random.shuffle(ids)
        test_size = min(len(ids) // 2, max_size)

        test_ids = ids[:test_size]
        train_ids = ids[test_size:]

        test_set.extend(test_ids)
        train_set.extend(train_ids)

    if verbose:
        print(f"Train: {len(train_set)}")
        print(f"Test:  {len(test_set)}")

    assert len(set(train_set).intersection(set(test_set))) == 0

    split_map = {i: "train" for i in train_set}
    split_map.update({i: "test" for i in test_set})
    df["split"] = df["id"].map(split_map)

    assert df["split"].isna().sum() == 0

    if save:
        # out_csv = join(BASE_PATH, "boneage_train_test.csv")
        out_csv = join(BASE_PATH, "boneage_train_test_peakfilter.csv")
        df.to_csv(out_csv, index=False)
        print(f"[Saved CSV] {out_csv}")

    # -----------------------------
    # visualization
    # -----------------------------
    if vis:
        fig, ax = plt.subplots(2, figsize=(6, 6), sharex=True)

        df_train = df[df["split"] == "train"]
        df_test = df[df["split"] == "test"]

        bins = range(df["boneage"].max() + 2)

        ax[0].hist(df_train["boneage"], bins=bins)
        ax[0].set_title(f"Train ({len(df_train)})")

        ax[1].hist(df_test["boneage"], bins=bins)
        ax[1].set_title(f"Test ({len(df_test)})")

        ax[1].set_xlabel("Bone Age (months)")

        plt.tight_layout()

        if save_fig:
            fig_dir = join(BASE_PATH, "figures")
            os.makedirs(fig_dir, exist_ok=True)

            # png_path = join(fig_dir, "boneage_train_test_distribution.png")
            # pdf_path = join(fig_dir, "boneage_train_test_distribution.pdf")
            png_path = join(fig_dir, "boneage_train_test_distribution_peakfilter.png")
            pdf_path = join(fig_dir, "boneage_train_test_distribution_peakfilter.pdf")

            plt.savefig(png_path, dpi=300)
            plt.savefig(pdf_path)
            print(f"[Saved Figure] {png_path}")
            print(f"[Saved Figure] {pdf_path}")

        plt.show()


if __name__ == "__main__":
    make_balanced_train_test(
        # csv_name="merged_train_val.csv",
        csv_name="merged_train_val_peak_filtered.csv",
        max_size=30,
        seed=666,
        verbose=True,
        vis=True,
        save=True,
        save_fig=True
    )












# ====================================================this is for boneage data processing===============to remove low data in peak region========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/home/ydubf/imbalanced-regression/BoneAge/merged_train_val.csv"

def plot_boneage_distribution(csv_path, bin_width=1):
    df = pd.read_csv(csv_path)

    ages = df["boneage"].values

    print(f"Total samples: {len(ages)}")
    print(f"Age range: [{ages.min()}, {ages.max()}]")

    bins = np.arange(ages.min(), ages.max() + bin_width, bin_width)

    plt.figure(figsize=(8, 4))
    plt.hist(ages, bins=bins)
    plt.xlabel("Bone Age (months)")
    plt.ylabel("Count")
    plt.title("BoneAge Overall Distribution (Train + Test)")
    plt.tight_layout()
    plt.savefig("/home/ydubf/imbalanced-regression/BoneAge/boneage_overall_distribution.pdf", dpi=300)
    plt.show()


# if __name__ == "__main__":
#     plot_boneage_distribution(CSV_PATH)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/home/ydubf/imbalanced-regression/BoneAge/merged_train_val.csv"

def detect_peak_and_sparse_labels(
    csv_path,
    window_size=10,          # 滑动窗口（months）
    peak_ratio=0.3,          # ≥ max_density * 0.7 视为 peak
    sparse_thresh=3,         # ≤ 3 的 label 视为“极小”
    plot=True,
):
    df = pd.read_csv(csv_path)

    ages = df["boneage"].astype(int)
    age_min, age_max = ages.min(), ages.max()

    # -----------------------
    # 1. 每个 label 的 count
    # -----------------------
    counts = ages.value_counts().sort_index()

    # 补齐 missing months
    all_ages = np.arange(age_min, age_max + 1)
    counts = counts.reindex(all_ages, fill_value=0)

    # -----------------------
    # 2. 滑动窗口估计密度
    # -----------------------
    window_density = []
    for a in all_ages:
        left = a - window_size // 2
        right = a + window_size // 2
        window_density.append(
            counts.loc[(counts.index >= left) & (counts.index <= right)].sum()
        )

    window_density = np.array(window_density)

    # -----------------------
    # 3. 定义 peak 区域
    # -----------------------
    max_density = window_density.max()
    peak_mask = window_density >= peak_ratio * max_density
    peak_ages = all_ages[peak_mask]

    peak_left, peak_right = peak_ages.min(), peak_ages.max()

    # -----------------------
    # 4. 在 peak 内找稀疏 label
    # -----------------------
    sparse_labels = [
        age for age in range(peak_left, peak_right + 1)
        if counts.loc[age] <= sparse_thresh
    ]

    # -----------------------
    # 5. 可视化
    # -----------------------
    if plot:
        plt.figure(figsize=(10, 4))
        plt.bar(all_ages, counts.values, width=1, alpha=0.7, label="Count")

        # peak 区域
        plt.axvspan(peak_left, peak_right, color="orange", alpha=0.2, label="Peak Region")

        # 稀疏 label
        plt.scatter(
            sparse_labels,
            [counts.loc[a] for a in sparse_labels],
            color="red",
            label="Sparse labels (candidate remove)",
            zorder=3
        )

        plt.xlabel("Bone Age (months)")
        plt.ylabel("Count")
        plt.title("BoneAge Distribution with Peak & Sparse Labels")
        plt.legend()
        plt.tight_layout()
        plt.savefig("/home/ydubf/imbalanced-regression/BoneAge/boneage_overall_distribution_filter.pdf", dpi=300)
        plt.show()

    return {
        "peak_range": (int(peak_left), int(peak_right)),
        "num_labels": len(counts),
        "num_sparse_in_peak": len(sparse_labels),
        "sparse_labels": sparse_labels,
        "counts": counts,
    }


# if __name__ == "__main__":
#     result = detect_peak_and_sparse_labels(
#         CSV_PATH,
#         window_size=10,
#         peak_ratio=0.25,
#         sparse_thresh=100,
#         plot=True
#     )

#     print("Peak range:", result["peak_range"])
#     print("Num sparse labels in peak:", result["num_sparse_in_peak"])
#     print("Example sparse labels:", result["sparse_labels"][:20])




import pandas as pd
import numpy as np

CSV_PATH = "/home/ydubf/imbalanced-regression/BoneAge/merged_train_val.csv"
OUT_CSV_PATH = "/home/ydubf/imbalanced-regression/BoneAge/merged_train_val_peak_filtered.csv"

def save_peak_filtered_csv(
    csv_path,
    out_csv_path,
    window_size=10,
    peak_ratio=0.25,
    sparse_thresh=100,
):
    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(csv_path)
    df["boneage"] = df["boneage"].astype(int)

    # -----------------------
    # Count per label
    # -----------------------
    ages = df["boneage"]
    age_min, age_max = ages.min(), ages.max()

    counts = ages.value_counts().sort_index()
    all_ages = np.arange(age_min, age_max + 1)
    counts = counts.reindex(all_ages, fill_value=0)

    # -----------------------
    # Sliding window density
    # -----------------------
    window_density = []
    for a in all_ages:
        left = a - window_size // 2
        right = a + window_size // 2
        window_density.append(
            counts.loc[(counts.index >= left) & (counts.index <= right)].sum()
        )
    window_density = np.array(window_density)

    # -----------------------
    # Peak detection
    # -----------------------
    max_density = window_density.max()
    peak_mask = window_density >= peak_ratio * max_density
    peak_ages = all_ages[peak_mask]

    peak_left, peak_right = int(peak_ages.min()), int(peak_ages.max())

    # -----------------------
    # Sparse labels in peak
    # -----------------------
    sparse_labels = {
        age for age in range(peak_left, peak_right + 1)
        if counts.loc[age] <= sparse_thresh
    }

    # -----------------------
    # Filter dataframe
    # -----------------------
    before_size = len(df)

    df_filtered = df[~df["boneage"].isin(sparse_labels)].copy()

    after_size = len(df_filtered)

    # -----------------------
    # Save
    # -----------------------
    df_filtered.to_csv(out_csv_path, index=False)

    # -----------------------
    # Report
    # -----------------------
    print("✅ Peak-aware filtering finished")
    print(f"Peak range           : [{peak_left}, {peak_right}]")
    print(f"Sparse thresh        : ≤ {sparse_thresh}")
    print(f"Num sparse labels    : {len(sparse_labels)}")
    print(f"Removed samples      : {before_size - after_size}")
    print(f"Remaining samples   : {after_size}")
    print(f"Saved to             : {out_csv_path}")

    return {
        "peak_range": (peak_left, peak_right),
        "sparse_labels": sorted(sparse_labels),
        "before": before_size,
        "after": after_size,
    }


# if __name__ == "__main__":
#     save_peak_filtered_csv(
#         CSV_PATH,
#         OUT_CSV_PATH,
#         window_size=10,
#         peak_ratio=0.25,
#         sparse_thresh=100,
#     )