# Data Directory

This directory is used to store datasets and related files.

## Directory Structure

Recommended dataset directory structure:



```
data/
├── mvtec/                  # MVTec AD 数据集
│   ├── bottle/
│   ├── cable/
│   └── ...
├── btad/                   # BTAD 数据集
│   ├── 01/
│   ├── 02/
│   └── 03/
├── visa/                   # VisA 数据集
├── dagm/                   # DAGM 数据集
├── mpdd/                   # MPDD 数据集
└── dtd/                    # DTD 纹理数据集（用于异常生成）
```

# Download the BTAD dataset
# Please visit the official website to obtain the download link


```bash
# Download the MVTec AD dataset
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz -C data/
```

### BTAD

```bash
# Download the BTAD dataset
# Please visit the official website to obtain the download link
```

### DTD Texture Dataset

```bash
# Download the DTD dataset (used for anomaly generation)
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xf dtd-r1.0.1.tar.gz -C data/
```

## Notes
- Ensure sufficient disk space (at least 50GB is recommended)

- Some datasets may require registration or application to download

- Verify dataset integrity after downloading

- SSD storage is recommended for faster training speed

## Data Preprocessing

After downloading, datasets may need preprocessing:

```bash
# Generate foreground masks
python scripts/generate_masks.py --dataset mvtec --data_root data/mvtec

# Verify dataset
python scripts/verify_dataset.py --dataset mvtec --data_root data/mvtec

```