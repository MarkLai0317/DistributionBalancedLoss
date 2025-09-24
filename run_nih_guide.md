# How to Run NIH Dataset

### Step 1. Download the NIH dataset

Download the dataset from Kaggle:
 👉 [NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data/data)

Assume the dataset root is **`DATAROOT`**.

### Step 2. Resample the dataset

The original dataset is very large, so resample it using:

```bash
python tools/create_longtail_dataset.py
```

⚠️ Update your data path between **lines 362–368** in the script.

Assume the resampled dataset root is **`LONGTAILDATAROOT`**.
 The resampled image list is saved in `LONGTAILDATAROOT/img_id.txt`, and the class split is saved in `LONGTAILDATAROOT/class_split.pkl`.

### Step 3. Organize images

The images are scattered under `images_xxx/images/`. Move them into `train/` and `test/` directories:

```bash
python tools/split_nih_images.py \
  --image-root DATAROOT \
  --train-txt LONGTAILDATAROOT/img_id.txt \
  --test-txt DATAROOT/test_list.txt \
  --output-root DATAROOT
```

### Step 4. Convert dataset format

Convert the long-tail dataset into a format compatible with the model:

```bash
python tools/df_to_csv_and_pkl.py \
  --df DATAROOT/Data_Entry_2017.csv \
  --image-root DATAROOT \
  --train-txt LONGTAILDATAROOT/img_id.txt \
  --test-txt DATAROOT/test_list.txt \
  --output-dir LONGTAILDATAROOT
```

The following files will be generated:

- `LONGTAILDATAROOT/train_annotations.pkl`
- `LONGTAILDATAROOT/train_data.csv`
- `LONGTAILDATAROOT/test_annotations.pkl`
- `LONGTAILDATAROOT/test_data.csv`
- `LONGTAILDATAROOT/class_freq.pkl`

### Step 5. Start training

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/nih/LT_resnet50_pfc_DB.py
```

### Step 6. Start testing

```bash
bash tools/dist_test.sh configs/nih/LT_resnet50_pfc_DB.py \
  work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test/epoch_8.pth 1
```