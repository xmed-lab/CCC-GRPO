import json

input_jsonl = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3000_linear_leq100.jsonl"   # 你的 jsonl 文件路径
output_json = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_conversation_sft_peak_compressed_3000_linear_leq100.json"   # 输出 json 文件路径

data = []

with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line))

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Converted {len(data)} samples from jsonl to json")