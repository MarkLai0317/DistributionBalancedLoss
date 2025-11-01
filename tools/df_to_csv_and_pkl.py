import os
import argparse
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


# --- Fixed label mapping (always same index order) ---
FIXED_LABELS = {
    'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Edema': 3, 'Effusion': 4,
    'Emphysema': 5, 'Fibrosis': 6, 'Hernia': 7, 'Infiltration': 8, 'Mass': 9,
    'No Finding': 10, 'Nodule': 11, 'Pleural_Thickening': 12, 'Pneumonia': 13, 'Pneumothorax': 14
}


def calculate_ratio(array: np.array, base_index: int, target_index: int):
    shared_count = np.sum((array[:, base_index] == 1) & (array[:, target_index] == 1))
    total_count = np.sum(array[:, base_index] == 1)
    return 0 if total_count == 0 else shared_count / total_count


def calculate_ratio_base_index(array: np.array, base_index: int):
    return np.array([calculate_ratio(array, base_index, i) for i in range(array.shape[1])])


def calculate_ratio_all(array: np.array):
    return np.array([calculate_ratio_base_index(array, i) for i in range(array.shape[1])])


def build_filename_index(image_root):
    """掃描所有子資料夾，建立 {filename: full_path}"""
    filename_to_path = {}
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.endswith(".png"):
                filename_to_path[f] = os.path.join(root, f)
    print(f"Indexed {len(filename_to_path)} images from {image_root}")
    return filename_to_path


def load_image_ids(txt_path):
    """載入圖片 ID 列表 (通用函數)"""
    with open(txt_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(ids)} images from {txt_path}")
    return ids


def generate_class_freq(df, label_columns, output_path):
    """生成 class_freq.pkl"""
    gt = df[label_columns].to_numpy()
    data = {
        "gt_labels": gt.tolist(),
        "class_freq": np.sum(gt == 1, axis=0),
        "neg_class_freq": np.sum(gt == 0, axis=0),
        "condition_prob": calculate_ratio_all(gt)
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved class frequencies to {output_path}")
    return data


def make_annotations(df, ids, label_columns, filename_to_path, split_name, output_path, csv_output_path):
    """產生 annotation list 並存成 pkl，同時輸出 csv"""
    annotations = []
    df_split = df[df["Image Index"].isin(ids)].copy()

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Building {split_name} annotations"):
        filename = row["Image Index"]
        image_path = filename_to_path.get(filename, None)

        if image_path is None or not os.path.exists(image_path):
            print(f"[WARNING] {split_name} image not found: {filename}")
            width, height = 0, 0
        else:
            try:
                image = Image.open(image_path)
                width, height = image.size
            except Exception as e:
                print(f"[WARNING] Cannot open {image_path}: {e}")
                width, height = 0, 0

        labels = np.array(row[label_columns].values, dtype=np.int32)

        annotation = {
            'filename': filename,
            'width': width,
            'height': height,
            'ann': {'labels': labels},
            'id': filename
        }
        annotations.append(annotation)

    # --- Save pkl ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(annotations, f)
    print(f"✅ Saved {split_name} annotations to {output_path}")

    # --- Save CSV (only ID + labels) ---
    csv_df = df_split[["Image Index"] + label_columns].rename(columns={"Image Index": "ID"})
    csv_df.to_csv(csv_output_path, index=False)
    print(f"✅ Saved {split_name} CSV to {csv_output_path}")

    return df_split


def main(args):
    # --- Step 1: Read CSV ---
    df = pd.read_csv(args.df)

    # --- Step 2: Add fixed multi-hot columns ---
    for label in FIXED_LABELS.keys():
        df[label] = df["Finding Labels"].apply(
            lambda x: 1 if label in x.split("|") else 0
        )
    label_columns = list(FIXED_LABELS.keys())
    print(f"Using fixed label mapping: {label_columns}")

    # --- Step 3: Build image index ---
    filename_to_path = build_filename_index(args.image_root)

    # --- Step 4: Load splits ---
    train_ids = load_image_ids(args.train_txt)
    eval_ids = load_image_ids(args.eval_txt) if args.eval_txt else []
    test_ids = load_image_ids(args.test_txt)

    # --- Step 5: Build train/eval/test annotations ---
    os.makedirs(args.output_dir, exist_ok=True)
    train_output = os.path.join(args.output_dir, "train_annotations.pkl")
    test_output = os.path.join(args.output_dir, "test_annotations.pkl")
    class_freq_path = os.path.join(args.output_dir, "class_freq.pkl")

    train_csv_output = os.path.join(args.output_dir, "train_data.csv")
    test_csv_output = os.path.join(args.output_dir, "test_data.csv")

    train_df = make_annotations(df, train_ids, label_columns, filename_to_path, "train", train_output, train_csv_output)
    test_df = make_annotations(df, test_ids, label_columns, filename_to_path, "test", test_output, test_csv_output)

    # Build eval annotations if eval_txt is provided
    if args.eval_txt and eval_ids:
        eval_output = os.path.join(args.output_dir, "eval_annotations.pkl")
        eval_csv_output = os.path.join(args.output_dir, "eval_data.csv")
        eval_df = make_annotations(df, eval_ids, label_columns, filename_to_path, "eval", eval_output, eval_csv_output)
        print(f"✅ Created eval split with {len(eval_ids)} images")

    # --- Step 6: Build class_freq.pkl (using train set only) ---
    generate_class_freq(train_df, label_columns, class_freq_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--image-root", required=True, help="Root folder (with images_001/, images_002/ ...)")
    parser.add_argument("--train-txt", required=True, help="Path to train split (txt only)")
    parser.add_argument("--eval-txt", help="Path to eval split (txt only, optional)")
    parser.add_argument("--test-txt", required=True, help="Path to test split (test_list.txt)")
    parser.add_argument("--output-dir", required=True, help="Directory to save all outputs")
    args = parser.parse_args()
    main(args)
