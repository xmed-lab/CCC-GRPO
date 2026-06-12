from os.path import join
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = "/home/ydubf/imbalanced-regression/poster_score"
IMG_DIR = join(BASE_PATH, "poster_downloads")
META_CSV = join(BASE_PATH, "poster.csv")


# --------------------------------------------------
# 1. Build meta: score → int label (score * 10)
# --------------------------------------------------
def build_poster_meta(img_dir):
    rows = []
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            score = float(fname.split("_")[0])
            label = int(round(score * 10))   # ⭐ 关键：统一成整数标签
            rows.append({
                "path": fname,
                "score": score,
                "label": label               # ⭐ 新增
            })
    return pd.DataFrame(rows)


# --------------------------------------------------
# 2. Balanced test split (AgeDB-style, no val)
# --------------------------------------------------
def make_balanced_testset(
    df,
    label_col="label",
    max_size=30,
    seed=666,
    verbose=True,
    vis=True,
    save=True,
):
    random.seed(seed)

    test_set = []

    for label in sorted(df[label_col].unique()):
        curr_df = df[df[label_col] == label]
        curr_data = curr_df["path"].values.tolist()
        random.shuffle(curr_data)

        # ⭐ follow AgeDB: at most 1/3 into test
        curr_size = min(len(curr_data) // 3, max_size)
        test_set += curr_data[:curr_size]

    if verbose:
        print(f"Test size: {len(test_set)}")

    test_set = set(test_set)
    df["split"] = df["path"].apply(
        lambda x: "test" if x in test_set else "train"
    )

    # if save:
    #     df.to_csv(join(BASE_PATH, "poster.csv"), index=False)
    #     print("✅ Saved poster.csv")

    if vis:
        visualize_splits(df)

    return df


# --------------------------------------------------
# 3. Visualization (integer label histogram)
# --------------------------------------------------
def visualize_splits(df):
    _, ax = plt.subplots(2, figsize=(7, 6), sharex=True)

    labels = np.arange(df["label"].min(), df["label"].max() + 1)

    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    ax[0].hist(df_train["label"], bins=labels)
    ax[0].set_title(f"Train ({len(df_train)})")
    ax[0].set_ylabel("Count")

    ax[1].hist(df_test["label"], bins=labels)
    ax[1].set_title(f"Test ({len(df_test)})")

    ax[1].set_xlabel("Poster Score")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(join(BASE_PATH, "poster_balanced_train_test.pdf"), dpi=300)
    plt.show()


# --------------------------------------------------
# 4. Main
# --------------------------------------------------
if __name__ == "__main__":
    df = build_poster_meta(IMG_DIR)

    make_balanced_testset(
        df,
        max_size=30,
        seed=666,
        vis=True,
        save=True
    )
    




import json
import os
import pandas as pd

BASE_PATH = "/home/ydubf/imbalanced-regression/poster_score"
IMG_DIR = os.path.join(BASE_PATH, "poster_downloads")
CSV_PATH = os.path.join(BASE_PATH, "poster.csv")

TRAIN_JSONL = os.path.join(BASE_PATH, "train.jsonl")
TEST_JSON = os.path.join(BASE_PATH, "test.json")

QUESTION = (
    "You are given a movie poster.\n\n"
    "Using only the visual cues in the poster, "
    "predict the movie’s IMDb rating score as accurately as possible.\n\n"
    "Return only one integer between 0 and 100 (IMDb score × 10)."
)


def build_train_test(csv_path):
    df = pd.read_csv(csv_path)

    train_samples = []
    test_samples = []

    uid = 0

    for _, row in df.iterrows():
        img_path = os.path.join(IMG_DIR, row["path"])
        label = int(row["label"])  # IMDb × 10

        if row["split"] == "train":
            train_samples.append({
                "id": uid,
                "image": img_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{QUESTION}"
                    },
                    {
                        "from": "gpt",
                        "value": str(label)
                    }
                ]
            })
            uid += 1

        elif row["split"] == "test":
            test_samples.append({
                "image": img_path,
                "problem": QUESTION,
                "solution": label
            })

    return train_samples, test_samples


# if __name__ == "__main__":
#     train_data, test_data = build_train_test(CSV_PATH)

#     # ---- save train.jsonl ----
#     with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
#         for sample in train_data:
#             f.write(json.dumps(sample) + "\n")

#     # ---- save test.json ----
#     with open(TEST_JSON, "w", encoding="utf-8") as f:
#         json.dump(test_data, f, indent=2)

#     print(f"✅ Saved train.jsonl ({len(train_data)} samples)")
#     print(f"✅ Saved test.json  ({len(test_data)} samples)")





# import json

# INPUT_JSONL = "/home/ydubf/imbalanced-regression/poster_score/train.jsonl"   # 你的 jsonl
# OUTPUT_JSON = "/home/ydubf/imbalanced-regression/poster_score/train_sft.json"    # 转换后的 json

# data = []

# with open(INPUT_JSONL, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         data.append(json.loads(line))

# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)

# print(f"✅ Converted {len(data)} samples from jsonl → json")



# import json

# INPUT_JSON = "/home/ydubf/imbalanced-regression/poster_score/test.json"          # 你现在的 test.json
# OUTPUT_JSON = "/home/ydubf/imbalanced-regression/poster_score/test_sft.json"     # 转换后的 json

# with open(INPUT_JSON, "r", encoding="utf-8") as f:
#     data = json.load(f)

# converted = []

# for idx, ex in enumerate(data):
#     converted.append({
#         "id": idx,
#         "image": ex["image"],
#         "conversations": [
#             {
#                 "from": "human",
#                 "value": "<image>" + ex["problem"]
#             },
#             {
#                 "from": "gpt",
#                 "value": str(ex["solution"])
#             }
#         ]
#     })

# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(converted, f, indent=2, ensure_ascii=False)

# print(f"✅ Converted {len(converted)} samples → {OUTPUT_JSON}")