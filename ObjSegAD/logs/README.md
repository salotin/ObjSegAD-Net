# Log Directory

This directory is used to store training and experiment log files.

## Directory Structure


```
logs/
├── training/ # Training logs
│ ├── mvtec/ # Training logs for MVTec dataset
│ ├── btad/ # Training logs for BTAD dataset
│ └── ...
├── tensorboard/ # TensorBoard logs
├── experiments/ # Experiment records
└── checkpoints/ # Model checkpoints
```


## Log Types

### Training Logs
- Detailed records of the training process  
- Loss function changes  
- Evaluation metrics  
- Model parameters  

### TensorBoard Logs
- Visualization of the training process  
- Loss curves  
- Learning rate changes  
- Model graph structure  

### Experiment Records
- Experiment configurations  
- Result summaries  
- Performance comparisons  

## Using TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# Access in browser
# http://localhost:6006

## Log Management

It is recommended to regularly clean up old log files to save disk space:

```bash
# Delete logs older than 30 days
find logs/ -name "*.log" -mtime +30 -delete

# Compress old checkpoints
tar -czf logs/old_checkpoints.tar.gz logs/checkpoints/old/

```