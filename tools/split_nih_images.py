import os
import shutil
import pickle
import argparse
from tqdm import tqdm


def build_filename_index(image_root):
    """掃描所有子資料夾，建立 {filename: full_path}"""
    filename_to_path = {}
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.endswith(".png"):
                filename_to_path[f] = os.path.join(root, f)
    print(f"Indexed {len(filename_to_path)} images from {image_root}")
    return filename_to_path


def load_train_ids(txt_path):
    """載入 train split (img_id.txt)"""
    ids = []
    with open(txt_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(ids)} train images from {txt_path}")
    return ids


def load_test_ids(txt_path):
    """載入 test split (test_list.txt)"""
    with open(txt_path, "r") as f:
        test_ids = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(test_ids)} test images from {txt_path}")
    return test_ids


def move_images(ids, split_name, filename_to_path, output_root, mode="symlink"):
    """把圖片搬到 split 資料夾"""
    split_dir = os.path.join(output_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for img in tqdm(ids, desc=f"Moving {split_name} images"):
        if img not in filename_to_path:
            print(f"[WARNING] {split_name} image not found: {img}")
            continue

        src = filename_to_path[img]
        dst = os.path.join(split_dir, img)

        if not os.path.exists(dst):  # 避免重複搬
            if mode == "copy":
                shutil.copy(src, dst)
            elif mode == "symlink":
                os.symlink(src, dst)
    print(f"✅ {split_name} images saved to {split_dir}")


def main(args):
    filename_to_path = build_filename_index(args.image_root)

    train_ids = load_train_ids(args.train_txt)
    test_ids = load_test_ids(args.test_txt)

    move_images(train_ids, "train", filename_to_path, args.output_root, args.mode)
    move_images(test_ids, "test", filename_to_path, args.output_root, args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, help="Root folder (with images_001/, images_002/ ...)")
    parser.add_argument("--train-txt", required=True, help="Path to img_id.txt")
    parser.add_argument("--test-txt", required=True, help="Path to test_list.txt")
    parser.add_argument("--output-root", required=True, help="Output folder for train/ and test/")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="Move mode: copy or symlink")
    args = parser.parse_args()
    main(args)