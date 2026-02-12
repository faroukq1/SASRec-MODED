#!/bin/bash

# Inference script for MovieLens 1M dataset with comprehensive metrics

if [ -z "$1" ]; then
    echo "Usage: ./inference_ml1m.sh <path_to_checkpoint>"
    echo "Example: ./inference_ml1m.sh ml-1m_default/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth"
    exit 1
fi

CKPT_PATH=$1

echo "Running inference on MovieLens 1M dataset..."
echo "Using checkpoint: $CKPT_PATH"
echo ""

cd python

python main.py \
    --dataset=ml-1m \
    --train_dir=default \
    --maxlen=200 \
    --device=cuda \
    --state_dict_path=$CKPT_PATH \
    --inference_only=true

echo ""
echo "Inference complete!"
