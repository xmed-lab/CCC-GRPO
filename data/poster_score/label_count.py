import json
from collections import defaultdict

# ====================== 你只需要改这里 ======================
# TRAIN_JSONL_PATH = "/home/ydubf/imbalanced-regression/poster_score/train.jsonl"  # 你的训练集路径
TRAIN_JSONL_PATH = "/home/ydubf/imbalanced-regression/agedb-dir/data/agedb_train.jsonl"  # 你的训练集路径

# OUTPUT_JSON_PATH = "/home/ydubf/imbalanced-regression/poster_score/label_distribution.json"
OUTPUT_JSON_PATH = "/home/ydubf/imbalanced-regression/agedb-dir/data/label_distribution.json"
MANY_THRESH = 100
MEDIUM_THRESH = 20
# ==========================================================

# 统计 label 数量
label_counter = defaultdict(int)

with open(TRAIN_JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        conversations = data["conversations"]
        # 找到 gpt 回答（就是 label）
        for turn in conversations:
            if turn["from"] == "gpt":
                label = int(turn["value"])
                label_counter[label] += 1

# 生成每个 label 的 info
label_info = {}
for label, cnt in label_counter.items():
    if cnt >= MANY_THRESH:
        region = "many"
    elif cnt >= MEDIUM_THRESH:
        region = "medium"
    else:
        region = "few"
    
    label_info[label] = {
        "count": cnt,
        "region": region
    }

# 保存 JSON
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "many_thresh": MANY_THRESH,
        "medium_thresh": MEDIUM_THRESH,
        "label_info": label_info
    }, f, indent=4, ensure_ascii=False)

print(f"✅ 生成完成！文件保存到: {OUTPUT_JSON_PATH}")
print(f"📊 总标签数: {len(label_info)}")