# ObjSegAD-Net: Region-aware pseudo-defect injection and dual-branch architecture for unsupervised industrial anomaly detection

An industrial anomaly detection framework based on DRAEM, supporting datasets such as MVTec AD.

## Features

- Dual-branch network based on DRAEM (Discriminatively Trained Reconstruction Embedding)  
- Integration of U2Net for foreground object segmentation and mask generation  
- Support for industrial anomaly detection datasets such as MVTec AD  
- Complete tools for training, inference, and visualization  

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train_mvtec.py   --data_root  --objects --resize -epochs --batch_size --lr -anomaly_source_dir 
```

### Inference test

```bash
python inference_testdata.py --model_path --test_dir
```


## File Description

- `model.py - DRAEM model definition

- `train.py - Training script

- `dataset.py - Dataset loader

- `losses_metrics.py - Loss functions and evaluation metrics

- `u2net.py - U2Net segmentation network

- `generate_mask.py - Mask generation tool

- `inference_testdata.py - Inference script

- `visualize_mvtec.py - Visualization of results


## License

This project is licensed under the MIT License - see the LICENSE
 file for details.