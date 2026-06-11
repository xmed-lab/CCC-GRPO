# import csv
# import json
# import os

# # csv_path = "/home/ydubf/imbalanced-regression/BoneAge/boneage_train_test.csv"
# csv_path = "/home/ydubf/imbalanced-regression/BoneAge/boneage_train_test_peakfilter.csv"
# output_json = "/home/ydubf/imbalanced-regression/BoneAge/test_conversation_from_boneage_peakfilter.json"

# results = []

# with open(csv_path, "r") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         if row["split"].strip().lower() == "test":
#             age = int(row["boneage"])

#             # ⭐ 原来只有 id，现在补上 .png
#             image_path = row["id"].strip()
#             if not image_path.endswith(".png"):
#                 image_path = image_path + ".png"
#             image_path = os.path.join("/home/ydubf/imbalanced-regression/BoneAge/boneage_resize", image_path)

#             sample = {
#                 "image": image_path,
#                 "problem": (
#                     "You are given a pediatric hand radiograph. Please assess the skeletal age based on the image.\n\n"
#                     "Task: Bone age estimation.\n\n"
#                     "Definition: Skeletal age is the estimated developmental age of the bones, measured in months.\n\n"
#                     "Constraints:\n"
#                     "- Minimum value: 0 months.\n"
#                     "- Maximum value: 216 months.\n"
#                     "- Step value: 1 month.\n\n"
#                     "Question: What is the skeletal age (in months) shown in this radiograph? Output a single integer number only."
#                 ),
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





# import csv
# import json
# import os

# # CSV_PATH = "/home/ydubf/imbalanced-regression/BoneAge/boneage_train_test.csv"              # 你的 agedb.csv 路径
# CSV_PATH = "/home/ydubf/imbalanced-regression/BoneAge/boneage_train_test_peakfilter.csv"
# # OUTPUT_JSONL = "/home/ydubf/imbalanced-regression/BoneAge/boneage_train.jsonl"  # 输出 JSONL 名称
# OUTPUT_JSONL = "/home/ydubf/imbalanced-regression/BoneAge/boneage_train_peakfilter.jsonl"  # 输出 JSONL 名称


# IMG_PREFIX = "/home/ydubf/imbalanced-regression/BoneAge/boneage_resize"          # 放图片的目录名，可按需修改

# jsonl = []

# with open(CSV_PATH, "r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)

#     idx = 0
#     for row in reader:
#         if row["split"].strip().lower() != "train":
#             continue

#         age = row["boneage"].strip()
#         # img_full_path = row["path"].strip()
#         # filename = os.path.basename(img_full_path)

#         img_rel_path = row["id"].strip()   # ← 保留 imdb_crop/01/...
#         if not img_rel_path.endswith(".png"):
#             img_rel_path = img_rel_path + ".png"

#         entry = {
#             "id": idx,
#             "image": os.path.join(IMG_PREFIX, img_rel_path),
#             "conversations": [
#                 {
#                     "from": "human",
#                     "value": "<image>You are given a pediatric hand radiograph. Please assess the skeletal age based on the image.\n\nTask: Bone age estimation.\n\nDefinition: Skeletal age is the estimated developmental age of the bones, measured in months.\n\nConstraints:\n- Minimum value: 0 months.\n- Maximum value: 216 months.\n- Step value: 1 month.\n\nQuestion: What is the skeletal age (in months) shown in this radiograph? Output a single integer number only."
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








import json

def jsonl_to_json(jsonl_path, output_json_path):
    data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Converted {len(data)} samples")
    print(f"Saved to: {output_json_path}")


# ===============================
# 使用示例
# ===============================
# jsonl_to_json(
#     jsonl_path="/home/ydubf/imbalanced-regression/BoneAge/boneage_train.jsonl",
#     output_json_path="/home/ydubf/imbalanced-regression/BoneAge/boneage_train_sft.json"
# )
# jsonl_to_json(
#     jsonl_path="/home/ydubf/imbalanced-regression/BoneAge/boneage_train_peakfilter.jsonl",
#     output_json_path="/home/ydubf/imbalanced-regression/BoneAge/boneage_train_peakfilter_sft.json"
# )




import json

def boneage_to_conversation_json(
    input_json_path,
    output_json_path,
    start_id=0
):
    """
    Convert BoneAge-style JSON to Qwen conversation-style JSON.
    """

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for idx, sample in enumerate(data):
        image_path = sample["image"]
        problem = sample["problem"]
        solution = sample["solution"]

        new_sample = {
            "id": start_id + idx,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>{problem}"
                },
                {
                    "from": "gpt",
                    "value": str(int(solution))
                }
            ]
        }

        new_data.append(new_sample)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print(f"✅ Converted {len(new_data)} samples")
    print(f"📁 Saved to: {output_json_path}")


# ===============================
# 使用示例
# ===============================
# boneage_to_conversation_json(
#     input_json_path="/home/ydubf/imbalanced-regression/BoneAge/test_conversation_from_boneage.json",
#     output_json_path="/home/ydubf/imbalanced-regression/BoneAge/test_conversation_from_boneage_sft.json",
#     start_id=0
# )
# boneage_to_conversation_json(
#     input_json_path="/home/ydubf/imbalanced-regression/BoneAge/test_conversation_from_boneage_peakfilter.json",
#     output_json_path="/home/ydubf/imbalanced-regression/BoneAge/test_conversation_from_boneage_peakfilter_sft.json",
#     start_id=0
# )








import os
from PIL import Image

def check_image_resolutions(folder):
    sizes = {}   # 记录每张图的分辨率

    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fname)
            try:
                with Image.open(path) as img:
                    sizes[fname] = img.size  # (W, H)
            except Exception as e:
                print(f"[Error] Cannot open {fname}: {e}")

    if not sizes:
        print("❌ No images found in folder.")
        return

    # 统计所有分辨率
    unique_sizes = set(sizes.values())

    print(f"📊 Total images: {len(sizes)}")
    print(f"📊 Unique resolutions found: {len(unique_sizes)}")

    # 如果全部一致
    if len(unique_sizes) == 1:
        print("✅ All images have the SAME resolution:", unique_sizes.pop())
        return

    print("⚠️ Resolutions NOT consistent!")
    print("📝 All unique resolutions:")
    for s in unique_sizes:
        print("   -", s)

    # 找最大、最小 (按像素面积排序)
    sorted_sizes = sorted(unique_sizes, key=lambda x: x[0] * x[1])

    min_size = sorted_sizes[0]
    max_size = sorted_sizes[-1]

    print("\n🔍 Smallest resolution:", min_size)
    for fname, s in sizes.items():
        if s == min_size:
            print("   →", fname)

    print("\n🔍 Largest resolution:", max_size)
    for fname, s in sizes.items():
        if s == max_size:
            print("   →", fname)

# ===== 使用方式 =====
# folder = "/home/ydubf/imbalanced-regression/BoneAge/boneage"
# check_image_resolutions(folder)







import os
import shutil
from PIL import Image

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




# import os
# import shutil
# from PIL import Image
# from tqdm import tqdm

# def resize_and_copy_all(
#     input_folder,
#     output_folder,
#     max_size=512
# ):
#     image_files = []

#     for root, _, files in os.walk(input_folder):
#         for fname in files:
#             if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 image_files.append(os.path.join(root, fname))

#     print(f"Found {len(image_files)} images.")

#     resized = copied = errors = 0

#     for in_path in tqdm(image_files, desc="Resizing images"):
#         rel_path = os.path.relpath(in_path, input_folder)
#         out_path = os.path.join(output_folder, rel_path)
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)

#         try:
#             with Image.open(in_path) as img:
#                 img = img.convert("RGB")
#                 w, h = img.size

#                 if max(w, h) > max_size:
#                     scale = max_size / max(w, h)
#                     new_w = int(w * scale)
#                     new_h = int(h * scale)
#                     img = img.resize((new_w, new_h), Image.BILINEAR)  # 🚀 快很多
#                     img.save(out_path)
#                     resized += 1
#                 else:
#                     shutil.copy2(in_path, out_path)
#                     copied += 1
#         except Exception as e:
#             errors += 1
#             print(f"[ERROR] {in_path}: {e}")

#     print("\n===== Resize Summary =====")
#     print(f"Resized : {resized}")
#     print(f"Copied  : {copied}")
#     print(f"Errors  : {errors}")
#     print("==========================")

# # # 使用
# input_folder = "/home/ydubf/imbalanced-regression/BoneAge/boneage"
# output_folder = "/home/ydubf/imbalanced-regression/BoneAge/boneage_resize"

# resize_and_copy_all(input_folder, output_folder, max_size=512)










import os
from PIL import Image
from collections import defaultdict

def check_image_resolutions_recursive(root_folder):
    sizes = {}              # filename -> (W, H)
    size_to_files = defaultdict(list)  # (W, H) -> [file paths]

    for root, _, files in os.walk(root_folder):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, fname)
                try:
                    with Image.open(path) as img:
                        size = img.size  # (W, H)
                        sizes[path] = size
                        size_to_files[size].append(path)
                except Exception as e:
                    print(f"[Error] Cannot open {path}: {e}")

    if not sizes:
        print("❌ No images found in folder.")
        return

    unique_sizes = list(size_to_files.keys())

    print(f"📊 Total images: {len(sizes)}")
    print(f"📊 Unique resolutions found: {len(unique_sizes)}")

    # 全一致
    if len(unique_sizes) == 1:
        print("✅ All images have the SAME resolution:", unique_sizes[0])
        return

    print("⚠️ Resolutions NOT consistent!")
    print("📝 All unique resolutions:")
    for s in sorted(unique_sizes, key=lambda x: x[0] * x[1]):
        print(f"   - {s}  (count={len(size_to_files[s])})")

    # 找最小 & 最大
    sorted_sizes = sorted(unique_sizes, key=lambda x: x[0] * x[1])

    min_size = sorted_sizes[0]
    max_size = sorted_sizes[-1]

    print("\n🔍 Smallest resolution:", min_size)
    for p in size_to_files[min_size][:5]:
        print("   →", p)
    if len(size_to_files[min_size]) > 5:
        print(f"   ... ({len(size_to_files[min_size])} files total)")

    print("\n🔍 Largest resolution:", max_size)
    for p in size_to_files[max_size][:5]:
        print("   →", p)
    if len(size_to_files[max_size]) > 5:
        print(f"   ... ({len(size_to_files[max_size])} files total)")



# root = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_crop"
# root = "/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/wiki_crop"
root = "/home/ydubf/imbalanced-regression/EchoNet-LVH-Keyframe/Key_frames"
check_image_resolutions_recursive(root)


















