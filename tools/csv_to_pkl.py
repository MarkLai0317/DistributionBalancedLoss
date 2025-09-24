import os
import re
import pickle
import pandas as pd
from PIL import Image
from tqdm import tqdm
from df_to_csv_and_pkl import calculate_ratio, calculate_ratio_base_index, calculate_ratio_all, build_filename_index, generate_class_freq

GROUP_CSV_PATTERN = re.compile(r"group\d+_(train|test)_data\.csv")

def make_annotations(df, label_columns, filename_to_path, output_path):
    """使用你原本版本生成 annotation pkl"""
    annotations = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building annotations"):
        filename = row["ID"]
        image_path = filename_to_path.get(filename, None)
        if image_path is None or not os.path.exists(image_path):
            width, height = 0, 0
        else:
            try:
                image = Image.open(image_path)
                width, height = image.size
            except:
                width, height = 0, 0

        labels = pd.Series(row[label_columns].values, dtype=int).to_numpy()
        annotation = {
            "filename": filename,
            "width": width,
            "height": height,
            "ann": {"labels": labels},
            "id": filename
        }
        annotations.append(annotation)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(annotations, f)
    print(f"✅ Saved annotations to {output_path}")
    return annotations


def process_group_csv(group_csv_path, class_split_pkl, image_root, output_dir):
    df = pd.read_csv(group_csv_path)
    filename_to_path = build_filename_index(image_root)

    # CSV 中的 label
    csv_labels = set(df.columns) - {"ID"}

    # 讀 class_split.pkl 並過濾
    with open(class_split_pkl, "rb") as f:
        class_split = pickle.load(f)
    filtered_split = {k: v & csv_labels for k, v in class_split.items()}

    # 根據 group CSV 名稱建立輸出資料夾
    group_name = os.path.splitext(os.path.basename(group_csv_path))[0]
    group_output_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_output_dir, exist_ok=True)

    # 生成 annotations.pkl
    annotations_path = os.path.join(group_output_dir, "annotations.pkl")
    make_annotations(df, list(csv_labels), filename_to_path, annotations_path)

    # 如果是 train，生成 class_freq.pkl
    if "train" in group_name:
        class_freq_path = os.path.join(group_output_dir, "class_freq.pkl")
        generate_class_freq(df, list(csv_labels), class_freq_path)

    # 保存 class_split.pkl（每個 group 一份）
    split_path = os.path.join(group_output_dir, "class_split.pkl")
    pickle.dump(filtered_split, open(split_path, "wb"))
    print(f"✅ Saved class split to {split_path}")


def main(group_csv_dir, class_split_pkl, image_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 使用正則找到 group CSV
    group_csv_files = [
        os.path.join(group_csv_dir, f)
        for f in os.listdir(group_csv_dir)
        if GROUP_CSV_PATTERN.match(f)
    ]

    for csv_file in group_csv_files:
        process_group_csv(csv_file, class_split_pkl, image_root, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_csv_dir", required=True, help="Directory containing grouped CSVs (group*_train/test_data.csv)")
    parser.add_argument("--class_split_pkl", required=True, help="Original class_split.pkl (head/middle/tail)")
    parser.add_argument("--image_root", required=True, help="Root folder containing images")
    parser.add_argument("--output_dir", required=True, help="Directory to save all output pkl files")
    args = parser.parse_args()

    main(args.group_csv_dir, args.class_split_pkl, args.image_root, args.output_dir)