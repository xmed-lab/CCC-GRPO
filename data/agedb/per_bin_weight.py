import json
import math
from collections import Counter

# -----------------------------
# 1. 解析 GT age
# -----------------------------
def extract_age(example):
    """
    从 conversations 中提取 gpt 的 age（字符串数字）
    """
    for turn in example.get("conversations", []):
        if turn.get("from") == "gpt":
            s = str(turn.get("value", "")).strip()
            if s.isdigit():
                return int(s)
    return None


# -----------------------------
# 2. 计算 age-bin 权重
# -----------------------------
def build_age_bin_weights(
    jsonl_path,
    num_bins=100,          # 0–99
    variant="log",         # "log" | "log2" | "inv"
    eps=1e-6,
    w_min=0.5,
    w_max=5.0,
):
    cnt = Counter()
    total = 0

    # 统计
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            age = extract_age(ex)
            if age is None:
                continue
            bin_id = max(0, min(num_bins - 1, age))
            # bin_id = max(1, min(num_bins - 1, age))
            cnt[bin_id] += 1
            total += 1

    if total == 0:
        raise RuntimeError("❌ No valid age labels found in jsonl.")

    # p_b
    p = {b: cnt.get(b, 0) / total for b in range(num_bins)}

    # raw weights
    w_raw = {}
    for b in range(num_bins):
        pb = p[b]
        if variant == "log":
            w_raw[b] = math.log(1.0 + 1.0 / (pb + eps))
        elif variant == "log2":
            w_raw[b] = math.log(1.0 + 1.0 / (pb + eps)) ** 2
        elif variant == "inv":
            w_raw[b] = 1.0 / (pb + eps)
        else:
            raise ValueError(f"Unknown variant: {variant}")

    # 归一化：E_p[w] = 1（非常重要）
    mean_w = sum(p[b] * w_raw[b] for b in range(num_bins))
    w = {b: w_raw[b] / (mean_w + 1e-12) for b in range(num_bins)}

    # clip 防爆
    # w = {b: max(w_min, min(w_max, w[b])) for b in range(num_bins)}
    w = {b: max(w_min, min(w_max, w[b])) for b in range(num_bins)}

    return w, cnt, total

    # w_shifted = {b + 1: w[b] for b in w}
    # return w_shifted, cnt, total


# -----------------------------
# 3. CLI / demo
# -----------------------------
if __name__ == "__main__":
    # jsonl_path = "/home/ydubf/imbalanced-regression/agedb-dir/data/agedb_train.jsonl"  # 改成你的路径
    jsonl_path = "/home/ydubf/imbalanced-regression/BoneAge/boneage_train.jsonl"  # 改成你的路径
    # jsonl_path = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3000_linear_leq100.jsonl"
    # jsonl_path ="/home/ydubf/imbalanced-regression/poster_score/train.jsonl"


    weights, counts, total = build_age_bin_weights(
        jsonl_path,
        num_bins=228,
        # num_bins=100,
        variant="log",      # ⭐ 推荐先用 log
        w_min=0.1,
        # w_max=5.0,
        w_max=2.0,
    )

    print(f"Total samples: {total}")
    print("Example bins:")
    for b in [0, 5, 18, 26, 40, 60, 80]:
        print(f"age {b:>2}: count={counts.get(b,0):>4}, weight={weights[b]:.3f}")

    # with open("/home/ydubf/imbalanced-regression/BoneAge/boneage_bin_weights.json", "w", encoding="utf-8") as f:
    with open("/home/ydubf/imbalanced-regression/BoneAge/boneage_bin_small_weights.json", "w", encoding="utf-8") as f:
    # with open("/home/ydubf/imbalanced-regression/poster_score/movie_bin_weights.json", "w", encoding="utf-8") as f:
    # with open("/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb-wiki-dir_bin_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    # print("✅ Saved age_bin_weights.json")