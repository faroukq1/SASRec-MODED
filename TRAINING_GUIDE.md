# MovieLens 1M Training with Comprehensive Metrics

This guide explains how to train the SASRec model on the MovieLens 1M dataset and get all the comprehensive metrics you need.

## Quick Start

### Training from scratch

```bash
./train_ml1m.sh
```

This will:
- Train on the MovieLens 1M dataset
- Save checkpoints every 20 epochs
- Display comprehensive metrics during training
- Save all metrics to a log file

### Running inference on a trained model

```bash
./inference_ml1m.sh <path_to_checkpoint>
```

Example:
```bash
./inference_ml1m.sh python/ml-1m_default/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth
```

## Metrics Computed

The model now computes the following metrics at K=10 and K=20:

### K=10 (Sequential Recommendation Standard)
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Recall@10**: Proportion of relevant items retrieved
- **Hit Rate@10**: Whether the relevant item appears in top-10
- **Precision@10**: Precision of recommendations
- **MRR@10**: Mean Reciprocal Rank
- **MAP@10**: Mean Average Precision

### K=20 (Graph Recommendation Standard)
- **NDCG@20**: Normalized Discounted Cumulative Gain
- **Recall@20**: Proportion of relevant items retrieved
- **Hit Rate@20**: Whether the relevant item appears in top-20
- **Precision@20**: Precision of recommendations
- **MRR@20**: Mean Reciprocal Rank
- **MAP@20**: Mean Average Precision

## Output Format

During training, you'll see output like:

```
epoch:20, time: 123.45(s)
  Valid - NDCG@10: 0.5234, HR@10: 0.7890
  Test  - NDCG@10: 0.5467, HR@10: 0.8123, Recall@10: 0.8123, Precision@10: 0.0812
```

At the end of training or during inference, you'll get:

```
============================================================
Best Test Set Metrics (based on validation performance):
============================================================

Metrics @ K=10 (Sequential Recommendation Standard):
  NDCG@10:      0.7750
  Recall@10:    0.9524
  Hit Rate@10:  0.9524
  Precision@10: 0.0952
  MRR@10:       0.7176
  MAP@10:       0.7176

Metrics @ K=20 (Graph Recommendation Standard):
  NDCG@20:      0.7829
  Recall@20:    0.9831
  Hit Rate@20:  0.9831
  Precision@20: 0.0492
  MRR@20:       0.7198
  MAP@20:       0.7198
============================================================
```

## Manual Training Command

If you want more control over the training parameters:

```bash
cd python
python main.py \
    --dataset=ml-1m \
    --train_dir=default \
    --maxlen=200 \
    --dropout_rate=0.2 \
    --device=cuda \
    --num_epochs=200 \
    --batch_size=128 \
    --lr=0.001 \
    --hidden_units=50 \
    --num_blocks=2 \
    --num_heads=1
```

### Key Parameters:
- `--dataset`: Dataset name (ml-1m for MovieLens 1M)
- `--train_dir`: Directory to save model checkpoints and logs
- `--maxlen`: Maximum sequence length
- `--dropout_rate`: Dropout rate for regularization
- `--device`: Device to use (cuda or cpu)
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--hidden_units`: Hidden dimension size
- `--num_blocks`: Number of transformer blocks
- `--num_heads`: Number of attention heads
- `--norm_first`: Use pre-normalization instead of post-normalization

## Log Files

All metrics are saved to: `python/ml-1m_default/log.txt`

The log file contains CSV format with all metrics for each evaluation epoch:
```
epoch,val_ndcg10,val_hr10,test_ndcg10,test_hr10,test_ndcg20,test_hr20,test_recall10,test_recall20,test_precision10,test_precision20,test_mrr10,test_mrr20,test_map10,test_map20
20,0.5234,0.7890,0.5467,0.8123,0.5678,0.8456,0.8123,0.8456,0.0812,0.0423,0.5234,0.5345,0.5234,0.5345
```

## Model Checkpoints

Model checkpoints are saved in: `python/ml-1m_default/`

Checkpoint naming format:
```
SASRec.epoch={epoch}.lr={lr}.layer={num_blocks}.head={num_heads}.hidden={hidden_units}.maxlen={maxlen}.pth
```

Example:
```
SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth
```

## System Requirements

- Python 3.6+
- PyTorch 1.6+
- CUDA-capable GPU (recommended) or CPU
- Required Python packages: numpy, torch

## Notes

- The model evaluates every 20 epochs during training
- Evaluation uses 100 negative samples per positive item
- For large datasets (>10000 users), evaluation samples 10000 random users
- Best model is saved based on validation set performance
