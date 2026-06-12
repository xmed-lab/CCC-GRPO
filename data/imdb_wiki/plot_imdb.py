import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

TRAIN_JSONL = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_leq100.jsonl"
OUT_FIG = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_distribution.png"


def plot_imdb_train_distribution(
    jsonl_path,
    max_age=100,
    save_path=None,
):
    age_counter = Counter()
    total = 0
    invalid = 0

    # -------------------------
    # Stream read jsonl
    # -------------------------
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
                convs = ex.get("conversations", [])
                for turn in convs:
                    if turn.get("from") == "gpt":
                        val = turn.get("value", "").strip()
                        if val.isdigit():
                            age = int(val)
                            if 0 <= age <= max_age:
                                age_counter[age] += 1
                                total += 1
                            else:
                                invalid += 1
                        else:
                            invalid += 1
                        break
            except Exception:
                invalid += 1

    # -------------------------
    # Prepare histogram
    # -------------------------
    ages = np.arange(0, max_age + 1)
    counts = np.array([age_counter.get(a, 0) for a in ages])

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(10, 4))
    plt.bar(ages, counts, width=1)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title(f"IMDb Train Age Distribution (N={total})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure saved to {save_path}")

    plt.show()

    # -------------------------
    # Stats
    # -------------------------
    print("===== IMDb Train Stats =====")
    print(f"Total valid samples : {total}")
    print(f"Invalid / skipped   : {invalid}")
    print(f"Min age             : {min(age_counter.keys())}")
    print(f"Max age             : {max(age_counter.keys())}")
    print("============================")

    return age_counter


# if __name__ == "__main__":
#     plot_imdb_train_distribution(
#         TRAIN_JSONL,
#         max_age=100,
#         save_path=OUT_FIG,
#     )







# import json
# import random
# from collections import defaultdict, Counter
# import numpy as np
# import matplotlib.pyplot as plt

# # ------------------------
# # Config
# # ------------------------
# IN_JSONL = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_leq100.jsonl"
# OUT_JSONL = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_1over3.jsonl"
# OUT_FIG = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_before_after_1over3.pdf"

# PEAK_LOW = 10
# PEAK_HIGH = 70
# PEAK_KEEP_RATIO = 1 / 3
# TAIL_KEEP_RATIO = 1.0
# MIN_KEEP_PER_LABEL = 5

# AGE_MIN = 0
# AGE_MAX = 100
# BINS = np.arange(AGE_MIN, AGE_MAX + 2, 1)

# random.seed(666)


# # ------------------------
# # Helper
# # ------------------------
# def extract_age(example):
#     for t in example.get("conversations", []):
#         if t.get("from") == "gpt":
#             v = t.get("value", "").strip()
#             if v.isdigit():
#                 return int(v)
#     return None


# # ------------------------
# # 1. Load & bucket
# # ------------------------
# buckets = defaultdict(list)
# all_ages = []

# with open(IN_JSONL, "r", encoding="utf-8") as f:
#     for line in f:
#         ex = json.loads(line)
#         age = extract_age(ex)
#         if age is None:
#             continue
#         buckets[age].append(ex)
#         all_ages.append(age)

# print(f"Loaded total samples: {len(all_ages)}")
# print(f"Unique ages: {len(buckets)}")

# # ------------------------
# # 2. Subsample (shape-preserving)
# # ------------------------
# kept = []
# kept_ages = []

# for age, samples in buckets.items():
#     n = len(samples)

#     if age < PEAK_LOW or age > PEAK_HIGH:
#         k = max(MIN_KEEP_PER_LABEL, int(n * TAIL_KEEP_RATIO))
#     else:
#         k = max(MIN_KEEP_PER_LABEL, int(n * PEAK_KEEP_RATIO))

#     random.shuffle(samples)
#     selected = samples[:k]

#     kept.extend(selected)
#     kept_ages.extend([age] * len(selected))

# print(f"Kept samples: {len(kept)} (~{len(kept)/len(all_ages):.2f})")

# # ------------------------
# # 3. Save new jsonl
# # ------------------------
# with open(OUT_JSONL, "w", encoding="utf-8") as f:
#     for ex in kept:
#         f.write(json.dumps(ex) + "\n")

# print(f"✅ Saved subsampled train set to {OUT_JSONL}")

# # ------------------------
# # 4. Plot before / after
# # ------------------------
# orig_counts = Counter(all_ages)
# kept_counts = Counter(kept_ages)

# orig_hist = np.array([orig_counts.get(a, 0) for a in range(AGE_MIN, AGE_MAX + 1)])
# kept_hist = np.array([kept_counts.get(a, 0) for a in range(AGE_MIN, AGE_MAX + 1)])

# fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# ax[0].bar(range(AGE_MIN, AGE_MAX + 1), orig_hist, width=1)
# ax[0].set_title(f"IMDb Train (Original, N={len(all_ages)})")
# ax[0].set_ylabel("Count")

# ax[1].bar(range(AGE_MIN, AGE_MAX + 1), kept_hist, width=1)
# ax[1].set_title(f"IMDb Train (Subsampled ~1/3, N={len(kept_ages)})")
# ax[1].set_xlabel("Age")
# ax[1].set_ylabel("Count")

# # Peak region hint（论文友好）
# ax[0].axvspan(PEAK_LOW, PEAK_HIGH, color="orange", alpha=0.15)
# ax[1].axvspan(PEAK_LOW, PEAK_HIGH, color="orange", alpha=0.15)

# plt.tight_layout()
# plt.savefig(OUT_FIG, dpi=300)
# plt.show()

# print(f"📊 Saved distribution figure to {OUT_FIG}")












import json
import random
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Config
# ------------------------
IN_JSONL = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_leq100.jsonl"
OUT_JSONL = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_1over3.jsonl"
OUT_FIG = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_before_after_1over3.pdf"

# 自动 peak 定义：count >= COUNT_THR 的区间（左右边界）
COUNT_THR = 100

# PEAK_KEEP_RATIO = 1 / 3      # 只对 count>=COUNT_THR 的 bins 下采样
PEAK_KEEP_RATIO = 2 / 7      # 只对 count>=COUNT_THR 的 bins 下采样
TAIL_KEEP_RATIO = 1.0        # count<COUNT_THR 的 bins 不动（全保留）
MIN_KEEP_PER_LABEL = 5       # 对被下采样的 bins 设一个下限（避免过小）

AGE_MIN = 0
AGE_MAX = 100

random.seed(666)


# ------------------------
# Helper
# ------------------------
def extract_age(example):
    for t in example.get("conversations", []):
        if t.get("from") == "gpt":
            v = str(t.get("value", "")).strip()
            # 兼容 "54", "54.0" 这类
            try:
                age = int(round(float(v)))
                return age
            except Exception:
                return None
    return None


# ------------------------
# 1. Load & bucket (filter abnormal gt)
# ------------------------
buckets = defaultdict(list)
all_ages = []

with open(IN_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        age = extract_age(ex)
        if age is None:
            continue
        # ⭐ 过滤异常 gt（0-100 之外）
        if not (AGE_MIN <= age <= AGE_MAX):
            continue

        buckets[age].append(ex)
        all_ages.append(age)

print(f"Loaded total samples (after filtering [{AGE_MIN},{AGE_MAX}]): {len(all_ages)}")
print(f"Unique ages: {len(buckets)}")

# ------------------------
# 2. Auto-detect peak boundaries by count>=COUNT_THR
# ------------------------
counts = Counter(all_ages)
ages_sorted = list(range(AGE_MIN, AGE_MAX + 1))

many_ages = [a for a in ages_sorted if counts.get(a, 0) >= COUNT_THR]

if len(many_ages) == 0:
    # 极端情况：没有任何 bin 达到阈值，那就不下采样
    PEAK_LOW, PEAK_HIGH = None, None
    print(f"[WARN] No bins have count >= {COUNT_THR}. Will keep all data.")
else:
    PEAK_LOW = min(many_ages)
    PEAK_HIGH = max(many_ages)
    print(f"Auto PEAK_LOW={PEAK_LOW}, PEAK_HIGH={PEAK_HIGH} (bins with count>={COUNT_THR})")
    print(f"Bin {PEAK_LOW} count : {counts.get(PEAK_LOW, 0)}")
    print(f"Bin {PEAK_HIGH} count: {counts.get(PEAK_HIGH, 0)}")

# ------------------------
# 3. Subsample (shape-preserving)
#   - bins with count < COUNT_THR: keep all (do NOT touch)
#   - bins with count >= COUNT_THR: keep 1/3 (with MIN_KEEP_PER_LABEL floor)
# ------------------------
kept = []
kept_ages = []

for age in ages_sorted:
    samples = buckets.get(age, [])
    n = len(samples)
    if n == 0:
        continue

    if n < COUNT_THR:
        k = int(n * TAIL_KEEP_RATIO)  # = n
    else:
        k = max(MIN_KEEP_PER_LABEL, int(n * PEAK_KEEP_RATIO))

    random.shuffle(samples)
    selected = samples[:k]
    kept.extend(selected)
    kept_ages.extend([age] * len(selected))

print(f"Kept samples: {len(kept)} (ratio={len(kept)/len(all_ages):.3f})")

# ------------------------
# 4. Save new jsonl
# ------------------------
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for ex in kept:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Saved subsampled train set to {OUT_JSONL}")

# ------------------------
# 5. Plot before / after
# ------------------------
orig_counts = Counter(all_ages)
kept_counts = Counter(kept_ages)

orig_hist = np.array([orig_counts.get(a, 0) for a in ages_sorted])
kept_hist = np.array([kept_counts.get(a, 0) for a in ages_sorted])

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax[0].bar(ages_sorted, orig_hist, width=1)
ax[0].set_title(f"IMDb Train (Original, N={len(all_ages)})")
ax[0].set_ylabel("Count")

ax[1].bar(ages_sorted, kept_hist, width=1)
ax[1].set_title(f"IMDb Train (Subsampled, N={len(kept_ages)})")
ax[1].set_xlabel("Age")
ax[1].set_ylabel("Count")

# Peak region hint（论文友好）
if PEAK_LOW is not None and PEAK_HIGH is not None:
    ax[0].axvspan(PEAK_LOW, PEAK_HIGH, color="orange", alpha=0.15)
    ax[1].axvspan(PEAK_LOW, PEAK_HIGH, color="orange", alpha=0.15)
    ax[1].text(
        0.02, 0.95,
        f"Downsample bins with count≥{COUNT_THR} by {PEAK_KEEP_RATIO:.3f}\nKeep bins <{COUNT_THR} unchanged",
        transform=ax[1].transAxes,
        va="top"
    )

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print(f"📊 Saved distribution figure to {OUT_FIG}")