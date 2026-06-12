#!/usr/bin/env python3
import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


DATASETS = ("agedb", "imdb_wiki", "imdb_movie", "boneage")


def iter_samples(shard_path):
    with tarfile.open(shard_path) as shard:
        members = shard.getmembers()
        metadata_members = [
            member for member in members if member.name.endswith(".json")
        ]
        for metadata_member in metadata_members:
            key = Path(metadata_member.name).stem
            metadata = json.load(shard.extractfile(metadata_member))
            image_member = next(
                member
                for member in members
                if Path(member.name).stem == key and not member.name.endswith(".json")
            )
            yield key, image_member.name, shard.extractfile(image_member).read(), metadata


def prepare_split(download_root, output_root, dataset, split):
    split_root = output_root / dataset
    image_root = split_root / "images" / split
    image_root.mkdir(parents=True, exist_ok=True)

    records = []
    shards = sorted((download_root / dataset / split).glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No shards found for {dataset}/{split}")

    for shard_path in shards:
        shard_id = shard_path.stem
        for key, member_name, image_bytes, metadata in iter_samples(shard_path):
            suffix = Path(member_name).suffix.lower()
            relative_image = Path("images") / split / f"{shard_id}_{key}{suffix}"
            (split_root / relative_image).write_bytes(image_bytes)
            records.append(
                {
                    "id": len(records),
                    "image": relative_image.as_posix(),
                    "problem": metadata["problem"].replace("<image>", "").strip(),
                    "solution": str(metadata["solution"]),
                }
            )

    if split == "train":
        output_path = split_root / "train.jsonl"
        with output_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "image": record["image"],
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": f"<image>{record['problem']}",
                                },
                                {"from": "gpt", "value": record["solution"]},
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        output_path = split_root / "test.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(records, output_file, ensure_ascii=False)

    print(f"{dataset}/{split}: {len(records)} samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="ChanganYao/DeepImbalancedRegressionForMLLMs",
    )
    parser.add_argument("--output-dir", default="hf_data")
    parser.add_argument("--download-dir", default=".cache/ccc_grpo_dataset")
    args = parser.parse_args()

    download_root = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.download_dir,
            allow_patterns=[f"{dataset}/**/*.tar" for dataset in DATASETS],
        )
    )
    output_root = Path(args.output_dir)
    for dataset in DATASETS:
        for split in ("train", "test"):
            prepare_split(download_root, output_root, dataset, split)


if __name__ == "__main__":
    main()
