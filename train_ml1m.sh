#!/bin/bash

# Training script for MovieLens 1M dataset with comprehensive metrics

echo "Starting training on MovieLens 1M dataset..."
echo "This will compute all metrics: NDCG, Recall, Hit Rate, Precision, MRR, and MAP at K=10 and K=20"
echo ""

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

echo ""
echo "Training complete! Check the metrics above and the log file in ml-1m_default/log.txt"
