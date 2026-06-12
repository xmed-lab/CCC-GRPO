# import pandas as pd

# def count_split(csv_path):
#     df = pd.read_csv(csv_path)

#     # 统计 split 列中 train/val/test 的数量
#     counts = df['split'].value_counts()

#     print("Dataset split statistics:")
#     print("-------------------------")
#     print(f"train: {counts.get('train', 0)}")
#     print(f"val:   {counts.get('val', 0)}")
#     print(f"test:  {counts.get('test', 0)}")

#     return counts

# # 运行
# # csv_path = "imbalanced-regression/agedb-dir/data/meta/agedb.csv"   # ← 这里是你上传的文件路径
# csv_path = "imdb-wiki-dir/data/imdb_wiki.csv"
# count_split(csv_path)





# import csv
# import json

# # csv_path = "imbalanced-regression/agedb-dir/data/agedb.csv"       # 你的 CSV 路径
# # output_json = "imbalanced-regression/agedb-dir/data/test_conversation_from_agedb.json"

# # csv_path = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_wiki.csv"       # 你的 CSV 路径
# # output_json = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/test_conversation_from_imdb.json"



# results = []

# with open(csv_path, "r") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         if row["split"].strip().lower() == "test":
#         # if row["split"].strip().lower() == "train":
#             age = int(row["age"])
#             image_path = row["path"].strip()

#             sample = {
#                 "image": image_path,
#                 "problem": "Age estimation: How old is the person in the image? Please answer with only a number.",
#                 "solution": age
#             }

#             results.append(sample)

# # 保存 JSON
# with open(output_json, "w") as f:
#     json.dump(results, f, indent=2)

# print(f"完成！共写入 {len(results)} 条测试样本 → {output_json}")





# import csv
# import json
# import os

# # ==== 路径修改成你的 ====
# CSV_PATH = "imbalanced-regression/agedb-dir/data/agedb.csv"                 # 你的 agedb.csv 文件
# OUTPUT_JSON = "imbalanced-regression/agedb-dir/data/agedb_test_data.json"   # 输出文件
# IMG_ROOT = ""                          # 若需要补全路径，例如 "/home/xxx/AgeDB/"

# # ==== 读取 CSV ====
# data = []
# with open(CSV_PATH, newline='', encoding='utf-8') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         age = int(row["age"])
#         path = row["path"]
#         split = row["split"]

#         # 只收集 test split（你可以换成 train/val/test）
#         if split.strip().lower() != "test":
#             continue

#         img_path = os.path.join(IMG_ROOT, path)

#         item = {
#             "id": len(data),  # 自动编号
#             "image": img_path,
#             "conversations": [
#                 {
#                     "from": "human",
#                     "value": "<image>Age estimation: How old is the person in the image? Please answer with only a number."
#                 },
#                 {
#                     "from": "gpt",
#                     "value": str(age)
#                 }
#             ]
#         }

#         data.append(item)

# # ==== 写 JSON ====
# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)

# print(f"Saved {len(data)} test samples → {OUTPUT_JSON}")





import csv
import json
import os

# CSV_PATH = "imbalanced-regression/agedb-dir/data/agedb.csv"              # 你的 agedb.csv 路径
# OUTPUT_JSONL = "imbalanced-regression/agedb-dir/data/agedb_train.jsonl"  # 输出 JSONL 名称

# CSV_PATH = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_wiki.csv"               # 你的 agedb.csv 路径
# OUTPUT_JSONL = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train.jsonl"  # 输出 JSONL 名称

# IMG_PREFIX = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/"          # 放图片的目录名，可按需修改

# jsonl = []

# with open(CSV_PATH, "r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)

#     idx = 0
#     for row in reader:
#         if row["split"].strip().lower() != "train":
#             continue

#         age = row["age"].strip()
#         # img_full_path = row["path"].strip()
#         # filename = os.path.basename(img_full_path)

#         img_rel_path = row["path"].strip()   # ← 保留 imdb_crop/01/...

#         entry = {
#             "id": idx,
#             # "image": os.path.join(IMG_PREFIX, filename),
#             # "image": filename,
#             "image": os.path.join(IMG_PREFIX, img_rel_path),
#             # "image": img_rel_path,
#             "conversations": [
#                 {
#                     "from": "human",
#                     "value": "<image>Age estimation: How old is the person in the image? Please answer with only a number."
#                 },
#                 {
#                     "from": "gpt",
#                     "value": age     # ground truth age
#                 }
#             ]
#         }

#         jsonl.append(entry)
#         idx += 1

# # 写入 JSONL
# with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
#     for item in jsonl:
#         fout.write(json.dumps(item, ensure_ascii=False) + "\n")

# print(f"Done! Train samples: {len(jsonl)} → {OUTPUT_JSONL}")






# import json

# # input_path = "imbalanced-regression/agedb-dir/data/test_conversation_from_agedb.json"       # 你的原始 JSON 文件
# # output_path = "imbalanced-regression/agedb-dir/data/test_conversation_from_agedb_sft.json"  # 输出文件

# # input_path = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/test_conversation_from_imdb.json"       # 你的原始 JSON 文件
# # output_path = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/test_conversation_from_imdb_sft.json"  # 输出文件

# input_path = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_sft.json"       # 你的原始 JSON 文件
# output_path = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_conversation_sft.json"  # 输出文件


# def convert(data):
#     new_data = []
#     for idx, item in enumerate(data):
#         img = item["image"]
#         gt = item["solution"]
#         # problem 是固定句式，所以你也可以直接取 item["problem"]
#         q = "<image>Age estimation: How old is the person in the image? Please answer with only a number."

#         new_item = {
#             "id": idx,
#             "image": img,
#             "conversations": [
#                 {"from": "human", "value": q},
#                 {"from": "gpt", "value": str(gt)}
#             ]
#         }
#         new_data.append(new_item)

#     return new_data


# if __name__ == "__main__":
#     with open(input_path, "r") as f:
#         data = json.load(f)

#     new_data = convert(data)

#     with open(output_path, "w") as f:
#         json.dump(new_data, f, indent=2)

#     print("转换完成:", output_path)





# import json
# import re

# input_path = "imbalanced-regression/agedb-dir/data/agedb_train.jsonl"      # 你的输入 JSONL 文件
# output_path = "imbalanced-regression/agedb-dir/data/output_fixed.jsonl"  # 输出 JSONL 文件

# def extract_age_from_filename(filename):
#     """从文件名里提取 GT，如 11706_OliviaHussey_31_f.jpg → 31"""
#     nums = re.findall(r"_([0-9]+)_", filename)
#     if len(nums) > 0:
#         return int(nums[-1])
#     return None

# with open(input_path, "r") as fin, open(output_path, "w") as fout:
#     new_id = 0
#     for line in fin:
#         line = line.strip()
#         if not line:
#             continue
        
#         item = json.loads(line)

#         img = item["image"]
#         gt = extract_age_from_filename(img)

#         # 统一的问题格式
#         question = "<image>Age estimation: How old is the person in the image? Please answer with only a number."

#         new_item = {
#             "id": new_id,
#             "image": img,
#             "conversations": [
#                 {"from": "human", "value": question},
#                 {"from": "gpt", "value": str(gt)}
#             ]
#         }
#         new_id += 1

#         fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

# print("转换完成，输出文件:", output_path)




# import json
# import re
# import os

# input_path = "imbalanced-regression/agedb-dir/data/output_fixed.jsonl"          # 输入 jsonl 文件
# output_path = "imbalanced-regression/agedb-dir/data/agedb_train_sft.json"   # 输出 json 文件（列表格式）

# def extract_age_from_filename(filename):
#     """从文件名中提取年龄，如 11706_OliviaHussey_31_f.jpg → 31"""
#     nums = re.findall(r"_([0-9]+)_", filename)
#     if len(nums) > 0:
#         return int(nums[-1])
#     return None

# data_out = []
# new_id = 0

# with open(input_path, "r") as fin:
#     for line in fin:
#         line = line.strip()
#         if not line:
#             continue

#         item = json.loads(line)

#         img = item["image"]
#         gt = extract_age_from_filename(img)

#         question = "<image>Age estimation: How old is the person in the image? Please answer with only a number."

#         new_item = {
#             "id": new_id,
#             "image": os.path.join("/home/ydubf/imbalanced-regression/agedb-dir/data/AgeDB/", img),
#             "conversations": [
#                 {"from": "human", "value": question},
#                 {"from": "gpt",   "value": str(gt)}
#             ]
#         }

#         data_out.append(new_item)
#         new_id += 1

# with open(output_path, "w") as fout:
#     json.dump(data_out, fout, indent=2, ensure_ascii=False)

# print("转换完成，输出文件:", output_path)





# import os
# from PIL import Image

# def check_image_resolutions(folder):
#     sizes = {}   # 记录每张图的分辨率

#     for fname in os.listdir(folder):
#         if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             path = os.path.join(folder, fname)
#             try:
#                 with Image.open(path) as img:
#                     sizes[fname] = img.size  # (W, H)
#             except Exception as e:
#                 print(f"[Error] Cannot open {fname}: {e}")

#     if not sizes:
#         print("❌ No images found in folder.")
#         return

#     # 统计所有分辨率
#     unique_sizes = set(sizes.values())

#     print(f"📊 Total images: {len(sizes)}")
#     print(f"📊 Unique resolutions found: {len(unique_sizes)}")

#     # 如果全部一致
#     if len(unique_sizes) == 1:
#         print("✅ All images have the SAME resolution:", unique_sizes.pop())
#         return

#     print("⚠️ Resolutions NOT consistent!")
#     print("📝 All unique resolutions:")
#     for s in unique_sizes:
#         print("   -", s)

#     # 找最大、最小 (按像素面积排序)
#     sorted_sizes = sorted(unique_sizes, key=lambda x: x[0] * x[1])

#     min_size = sorted_sizes[0]
#     max_size = sorted_sizes[-1]

#     print("\n🔍 Smallest resolution:", min_size)
#     for fname, s in sizes.items():
#         if s == min_size:
#             print("   →", fname)

#     print("\n🔍 Largest resolution:", max_size)
#     for fname, s in sizes.items():
#         if s == max_size:
#             print("   →", fname)

# # ===== 使用方式 =====
# folder = "/home/ydubf/imbalanced-regression/agedb-dir/data/AgeDB"
# check_image_resolutions(folder)







# import os
# import shutil
# from PIL import Image

# def resize_and_copy_all(
#     input_folder, 
#     output_folder, 
#     max_size=512
# ):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     resized = 0
#     copied = 0

#     for fname in os.listdir(input_folder):
#         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue

#         in_path = os.path.join(input_folder, fname)
#         out_path = os.path.join(output_folder, fname)

#         try:
#             with Image.open(in_path) as img:
#                 w, h = img.size

#                 # 如果大于 max_size → 缩放
#                 if max(w, h) > max_size:
#                     scale = max_size / max(w, h)
#                     new_w = int(w * scale)
#                     new_h = int(h * scale)

#                     img_resized = img.resize((new_w, new_h), Image.LANCZOS)
#                     img_resized.save(out_path)
#                     resized += 1
#                 else:
#                     # 小图直接 copy，不修改
#                     shutil.copy(in_path, out_path)
#                     copied += 1

#         except Exception as e:
#             print(f"[ERROR] Cannot process {fname}: {e}")

#     print("\n===== Resize Summary =====")
#     print(f"Resized large images: {resized}")
#     print(f"Copied small images: {copied}")
#     print(f"Output saved in: {output_folder}")
#     print("==========================")

# # 使用
# input_folder = "/home/ydubf/imbalanced-regression/agedb-dir/data/AgeDB_origin"
# output_folder = "/home/ydubf/imbalanced-regression/agedb-dir/data/AgeDB"

# resize_and_copy_all(input_folder, output_folder, max_size=512)










# import os
# from PIL import Image
# from collections import defaultdict

# def check_image_resolutions_recursive(root_folder):
#     sizes = {}              # filename -> (W, H)
#     size_to_files = defaultdict(list)  # (W, H) -> [file paths]

#     for root, _, files in os.walk(root_folder):
#         for fname in files:
#             if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 path = os.path.join(root, fname)
#                 try:
#                     with Image.open(path) as img:
#                         size = img.size  # (W, H)
#                         sizes[path] = size
#                         size_to_files[size].append(path)
#                 except Exception as e:
#                     print(f"[Error] Cannot open {path}: {e}")

#     if not sizes:
#         print("❌ No images found in folder.")
#         return

#     unique_sizes = list(size_to_files.keys())

#     print(f"📊 Total images: {len(sizes)}")
#     print(f"📊 Unique resolutions found: {len(unique_sizes)}")

#     # 全一致
#     if len(unique_sizes) == 1:
#         print("✅ All images have the SAME resolution:", unique_sizes[0])
#         return

#     print("⚠️ Resolutions NOT consistent!")
#     print("📝 All unique resolutions:")
#     for s in sorted(unique_sizes, key=lambda x: x[0] * x[1]):
#         print(f"   - {s}  (count={len(size_to_files[s])})")

#     # 找最小 & 最大
#     sorted_sizes = sorted(unique_sizes, key=lambda x: x[0] * x[1])

#     min_size = sorted_sizes[0]
#     max_size = sorted_sizes[-1]

#     print("\n🔍 Smallest resolution:", min_size)
#     for p in size_to_files[min_size][:5]:
#         print("   →", p)
#     if len(size_to_files[min_size]) > 5:
#         print(f"   ... ({len(size_to_files[min_size])} files total)")

#     print("\n🔍 Largest resolution:", max_size)
#     for p in size_to_files[max_size][:5]:
#         print("   →", p)
#     if len(size_to_files[max_size]) > 5:
#         print(f"   ... ({len(size_to_files[max_size])} files total)")



# # root = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_crop"
# root = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/wiki_crop"
# check_image_resolutions_recursive(root)







# import json
# import random

# def sample_jsonl(
#     input_path,
#     output_path,
#     keep_ratio=0.1,
#     seed=42
# ):
#     random.seed(seed)

#     kept = 0
#     total = 0

#     with open(input_path, "r", encoding="utf-8") as fin, \
#          open(output_path, "w", encoding="utf-8") as fout:

#         for line in fin:
#             total += 1
#             if random.random() < keep_ratio:
#                 fout.write(line)
#                 kept += 1

#     print(f"Done! Kept {kept}/{total} samples "
#           f"({kept/total:.2%})")

# if __name__ == "__main__":
#     sample_jsonl(
#         input_path="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train.jsonl",
#         output_path="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_1of10.jsonl",
#         keep_ratio=0.1
#     )




import json
import random
import numpy as np
from tqdm import tqdm

def filter_main_samples_jsonl(
    input_jsonl,
    output_jsonl,
    low_percentile=10,
    high_percentile=90,
    main_keep_ratio=0.3,
    seed=42,
):
    """
    Filter train jsonl to reduce main samples while keeping extreme samples intact.

    Args:
        input_jsonl (str): path to original train.jsonl
        output_jsonl (str): path to filtered train.jsonl
        low_percentile (float): percentile defining lower extreme
        high_percentile (float): percentile defining upper extreme
        main_keep_ratio (float): ratio to keep for main samples (e.g., 0.3 keeps 30%)
        seed (int): random seed
    """

    random.seed(seed)

    # -----------------------------
    # Load all samples
    # -----------------------------
    samples = []
    labels = []

    with open(input_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            y = float(data["conversations"][-1]["value"])
            samples.append(data)
            labels.append(y)

    labels = np.array(labels)

    # -----------------------------
    # Compute extreme thresholds
    # -----------------------------
    low_thr = np.percentile(labels, low_percentile)
    high_thr = np.percentile(labels, high_percentile)

    print(f"[INFO] Extreme low <= {low_thr:.2f}, extreme high >= {high_thr:.2f}")

    # -----------------------------
    # Split samples
    # -----------------------------
    extreme_samples = []
    main_samples = []

    for s, y in zip(samples, labels):
        if y <= low_thr or y >= high_thr:
            extreme_samples.append(s)
        else:
            main_samples.append(s)

    # -----------------------------
    # Downsample main samples
    # -----------------------------
    keep_main_num = int(len(main_samples) * main_keep_ratio)
    main_samples = random.sample(main_samples, keep_main_num)

    final_samples = extreme_samples + main_samples
    random.shuffle(final_samples)

    # -----------------------------
    # Save
    # -----------------------------
    with open(output_jsonl, "w") as f:
        for s in final_samples:
            f.write(json.dumps(s) + "\n")

    print("========== Summary ==========")
    print(f"Original samples : {len(samples)}")
    print(f"Extreme samples  : {len(extreme_samples)} (kept all)")
    print(f"Main samples     : {len(main_samples)} (kept {main_keep_ratio*100:.1f}%)")
    print(f"Final samples    : {len(final_samples)}")
    print(f"Saved to         : {output_jsonl}")


# filter_main_samples_jsonl(
#     input_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_1of5.jsonl",
#     output_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_1of5_filtered.jsonl",
#     low_percentile=10,
#     high_percentile=90,
#     main_keep_ratio=0.50,  # main 只保留 25%
# )









import json
import numpy as np
from collections import defaultdict
import random

def load_ages(jsonl_path):
    ages = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            age = int(data["conversations"][1]["value"])
            ages.append(age)
    return np.array(ages)


def compress_histogram(
    hist,
    n_target_peak,
    gamma=2.0,
    low_frac=0.05
):
    """
    hist: dict {bin_value: count}
    """
    counts = np.array(list(hist.values()))
    n_max = counts.max()
    n_low = low_frac * n_max

    print(n_low)

    new_hist = {}
    for b, n in hist.items():
        if n <= n_low:
            new_hist[b] = int(n)
        else:
            ratio = (n - n_low) / (n_max - n_low)
            n_new = n_low + (n_target_peak - n_low) * (ratio ** gamma)
            new_hist[b] = int(round(n_new))
    return new_hist


def filter_jsonl_by_hist(
    jsonl_path,
    output_path,
    new_hist
):
    # 先按 age 分组
    age_to_samples = defaultdict(list)

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            age = int(data["conversations"][1]["value"])
            age_to_samples[age].append(line)

    # 按目标 bin 数量采样
    selected_lines = []
    for age, samples in age_to_samples.items():
        target_n = new_hist.get(age, 0)
        if target_n >= len(samples):
            selected_lines.extend(samples)
        else:
            selected_lines.extend(random.sample(samples, target_n))
    print(len(selected_lines))

    # 写回
    with open(output_path, "w") as f:
        for line in selected_lines:
            f.write(line)


ages = load_ages("/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train.jsonl")

# bin = 1 histogram
hist = dict(zip(*np.unique(ages, return_counts=True)))

new_hist = compress_histogram(
    hist,
    # n_target_peak=1500,  # 你图里的目标
    # n_target_peak=3500,  # 你图里的目标
    # gamma=2.5,
    # low_frac=0.08
    n_target_peak=2000,  # 你图里的目标
    gamma=1,
    low_frac=0.1
)

filter_jsonl_by_hist(
    "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train.jsonl",
    "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_2000_linear.jsonl",
    new_hist
)




# ========================================将大于100的target样本去除=====================================

import json
import re

def filter_jsonl_by_value(
    input_jsonl,
    output_jsonl,
    threshold=100
):
    kept = 0
    removed = 0
    bad_lines = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for line_id, line in enumerate(fin):
            try:
                data = json.loads(line)

                val = None
                for turn in data.get("conversations", []):
                    if turn.get("from") == "gpt":
                        match = re.search(r"-?\d+(\.\d+)?", str(turn.get("value", "")))
                        if match:
                            val = float(match.group())
                        break

                # 如果没解析出数值，默认保留（更安全）
                if val is None or val <= threshold:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    removed += 1

            except Exception as e:
                bad_lines += 1
                print(f"[Warning] Line {line_id} skipped: {e}")

    print("✅ JSONL filtering finished")
    print(f"📄 Kept samples: {kept}")
    print(f"🗑️  Removed samples (value > {threshold}): {removed}")
    print(f"❌ Bad lines: {bad_lines}")


# filter_jsonl_by_value(
#     # input_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train.jsonl",
#     # output_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_leq100.jsonl",
#     # input_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3500.jsonl",
#     # output_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3500_leq100.jsonl",
#     input_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3000_linear.jsonl",
#     output_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3000_linear_leq100.jsonl",
#     # input_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed.jsonl",
#     # output_jsonl="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_leq100.jsonl",
#     threshold=100
# )




# import json

# def filter_json_by_solution(
#     input_json,
#     output_json,
#     threshold=100
# ):
#     with open(input_json, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     kept_data = []
#     removed = 0
#     invalid = 0

#     for item in data:
#         try:
#             sol = float(item["solution"])
#             if sol <= threshold:
#                 kept_data.append(item)
#             else:
#                 removed += 1
#         except Exception:
#             invalid += 1
#             kept_data.append(item)  # 无法解析的，保留更安全

#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(kept_data, f, ensure_ascii=False, indent=2)

#     print("✅ JSON filtering finished")
#     print(f"📄 Kept samples: {len(kept_data)}")
#     print(f"🗑️  Removed samples (solution > {threshold}): {removed}")
#     print(f"❌ Invalid solution fields: {invalid}")

# filter_json_by_solution(
#     input_json="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/test_conversation_from_imdb.json",
#     output_json="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/test_conversation_from_imdb_leq100.json",
#     threshold=100
# )




# import json
# import re

# def filter_json_conversations_by_value(
#     input_json,
#     output_json,
#     threshold=100
# ):
#     with open(input_json, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     kept = []
#     removed = 0
#     invalid = 0

#     for idx, item in enumerate(data):
#         try:
#             val = None
#             for turn in item.get("conversations", []):
#                 if turn.get("from") == "gpt":
#                     match = re.search(r"-?\d+(\.\d+)?", str(turn.get("value", "")))
#                     if match:
#                         val = float(match.group())
#                     break

#             # 如果没解析到数值：保留（更安全）
#             if val is None or val <= threshold:
#                 kept.append(item)
#             else:
#                 removed += 1

#         except Exception as e:
#             print(f"[Warning] Item {idx} failed to parse: {e}")
#             invalid += 1
#             kept.append(item)  # 出错的也保留，避免误删

#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(kept, f, ensure_ascii=False, indent=2)

#     print("✅ JSON filtering finished")
#     print(f"📄 Original samples: {len(data)}")
#     print(f"✅ Kept samples: {len(kept)}")
#     print(f"🗑️  Removed samples (value > {threshold}): {removed}")
#     print(f"❌ Invalid samples (kept): {invalid}")

# filter_json_conversations_by_value(
#     input_json="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/test_conversation_from_imdb_sft.json",
#     output_json="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/test_conversation_from_imdb_sft_leq100.json",
#     threshold=100
# )
            




import json
import random

def jsonl_to_json_list_and_shuffle(
    jsonl_path,
    output_json_path,
    reset_id=True,
    shuffle=True,
    seed=42
):
    """
    Convert jsonl (one JSON per line) to a shuffled JSON list.

    Args:
        jsonl_path: path to input .jsonl file
        output_json_path: path to output .json file
        reset_id: if True, reassign id = 0,1,2,...
        shuffle: whether to shuffle samples
        seed: random seed for reproducibility
    """
    data_list = []
    bad_lines = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data_list.append(item)
            except Exception as e:
                bad_lines += 1
                print(f"[Warning] Line {idx} skipped: {e}")

    # -------------------------------
    # Shuffle
    # -------------------------------
    if shuffle:
        random.seed(seed)
        random.shuffle(data_list)

    # -------------------------------
    # Reset id
    # -------------------------------
    if reset_id:
        for i, item in enumerate(data_list):
            item["id"] = i

    # -------------------------------
    # Save
    # -------------------------------
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted samples: {len(data_list)}")
    print(f"🔀 Shuffled: {shuffle} (seed={seed})")
    print(f"⚠️  Skipped bad lines: {bad_lines}")
    print(f"💾 Saved to: {output_json_path}")


# jsonl_to_json_list_and_shuffle(
#     # jsonl_path="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_leq100.jsonl",
#     # output_json_path="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_conversation_sft_peak_compressed_leq100.json",
#     jsonl_path="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3500_leq100.jsonl",
#     output_json_path="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_conversation_sft_peak_compressed_3500_leq100.json",
#     shuffle=True,
#     seed=42
#     # reset_id=True
# )







import os
from PIL import Image

# =========================
# 配置路径
# =========================
INPUT_DIR = "/home/ydubf/imbalanced-regression/diabetic-retina/train"
OUTPUT_DIR = "/home/ydubf/imbalanced-regression/diabetic-retina/train_resized"
TARGET_SIZE = (450, 300)  # (width, height)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Resize 所有图片
# =========================
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)

    try:
        with Image.open(in_path) as img:
            img = img.convert("RGB")
            img_resized = img.resize(TARGET_SIZE, Image.BILINEAR)
            img_resized.save(out_path, quality=95)

        print(f"[OK] {fname}")

    except Exception as e:
        print(f"[FAIL] {fname}: {e}")

print("✅ All images resized.")