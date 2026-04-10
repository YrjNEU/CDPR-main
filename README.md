  # CDPR: Cross-modal Diffusion with Polarization for Reliable Monocular Depth Estimation
**IEEE TMM 2026**

![pipeline](https://github.com/user-attachments/assets/f58294e4-ccfc-4f63-a711-3c6748539d28)

This repository contains the source code of our paper **CDPR: Cross-modal Diffusion with Polarization for Reliable Monocular Depth Estimation**.

- `CDPR-original` corresponds to the `depth` version in this repository.
- `CDPR-accelerated` corresponds to the `depth_v1` version in this repository.


## Environment


The project was implemented with:

- Ubuntu 22.04
- NVIDIA RTX 4090
- Python 3.10
- CUDA 11.7

Recommended setup with Conda/Mamba:

```bash
conda env create -f environment.yaml
conda activate cdpr
pip install -r requirements.txt
pip install -r requirements+.txt
pip install -r requirements++.txt
```

If `xformers` is not installed automatically, install the version matching your local CUDA/PyTorch environment.

## Data Preprocessing

### 1. HyperPol

You can download [original Hypersim Dataset](https://github.com/apple/ml-hypersim) as follow:

```bash
python script/dataset_preprocess/hypersim/dowmload_hypersim.py \
  --delete_archive_after_decompress
  --downloads_dir /path/to/temp_archive
  --decompress_dir /path/to/hypersim/raw_data
```

Then generate DoLP/AoLP and resplit the dataset:

```bash
python script/dataset_preprocess/hypersim/preprocess_hypersim.py \
  --split_csv /path/to/metadata_images_split_scene_v1.csv \
  --dataset_dir /path/to/hypersim/raw_data \
  --output_dir ${BASE_DATA_DIR}/hypersim/processed
```


### 2. HAMMER
You need to download [raw HAMMER dataset](https://github.com/Junggy/HAMMER-dataset) first, then preprocess the raw dataset as follows:

```bash
python script/dataset_preprocess/hammer/preprocess_hammer.py \
  --source_folder /path/to/hammer/raw_data \
  --output_folder ${BASE_DATA_DIR}/hammer/processed_data
```

### 3. HouseCat6D
For raw HouseCat6D dataset, we provide two options for download:
#### Option 1: Automatic Download (Recommended)
```bash
python script/dataset_preprocess/housecat6d/download.py \
  --output_root ${BASE_DATA_DIR}/HouseCat6D
```
#### Option 2: Manual Download
You can also download the dataset from the [official website](https://sites.google.com/view/housecat6d/dataset).Please download the Test Set.

```bash
python script/dataset_preprocess/housecat6d/preprocess.py \
  --source_root /path/to/test_scene \
  --output_root ${BASE_DATA_DIR}/HouseCat6D/processed
```
#### Note: The evaluation results on HouseCat6D are not included in the main paper due to the limited Ground Truth, see more details in the supplementary material.
## Training

Set environment parameters for the data directory:

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR  # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}`

### CDPR-original

```bash
python train.py --config config/train_cdpr_depth.yaml
```

### CDPR-accelerated

```bash
python train_depth_v1.py --config config/train_cdpr_depth_v1.yaml
```

## Inference
Our trained checkpoint can be downloaded as follows:
```bash
git clone https://huggingface.co/RongJia1/CDPR-checkpoint
```
You can directly test the provided sample set in the current repository without training:
### CDPR-original
```bash
python run.py \
    --checkpoint CDPR-checkpoint/checkpoint/depth/CDPR-CNN \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_dir input \
    --output_dir output
```

### CDPR-accelerated

```bash
python run.py \
    --checkpoint CDPR-checkpoint/checkpoint/depth/CDPR-CNNv1 \
    --denoise_steps 4 \
    --ensemble_size 1 \
    --input_dir input \
    --output_dir output
```

## Evaluation

### CDPR-original

```bash
python evaluate.py \
  --config config/eval.yaml \
  --checkpoint CDPR-checkpoint/checkpoint/depth/CDPR-CNN \
  --base_data_dir ${BASE_DATA_DIR} \
  --output_dir output/eval_cdpr_original
```

### CDPR-accelerated

```bash
python evaluate.py \
  --config config/eval_v1.yaml \
  --checkpoint CDPR-checkpoint/checkpoint/depth/CDPR-CNNv1 \
  --base_data_dir ${BASE_DATA_DIR} \
  --output_dir output/eval_cdpr_accelerated
```

## Notes

- This repository does not include trained model weights or the Stable Diffusion base checkpoint.
- To reproduce experiments directly, place your own `CDPR-polCNN` or `CDPR-polCNNv1` checkpoint under `checkpoint/depth/`, or pass the real checkpoint path on the command line.
