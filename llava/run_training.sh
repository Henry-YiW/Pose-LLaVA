#!/bin/bash

# This script processes .mat files and trains the Pose-LLaVA model

# Define paths
MAT_DIR="./labels"  # Directory containing .mat files
PROCESSED_DIR="./llava/processed_data"  # Directory for processed data
OUTPUT_DIR="./llava/pose_llava_data"  # Directory for training data
MODEL_OUTPUT_DIR="./llava/checkpoints/pose-llava-7b"  # Directory for trained model

# Create directories
mkdir -p $PROCESSED_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $MODEL_OUTPUT_DIR

# Step 1: Process MAT files
echo "Processing MAT files..."
python process_mat_files.py \
    --input-dir $MAT_DIR \
    --output-path $PROCESSED_DIR/pose_data.npy \
    --split-ratio 0.8

# Step 2: Create training data
echo "Creating training data..."
python create_training_data.py \
    --train-data $PROCESSED_DIR/pose_data_train.npy \
    --val-data $PROCESSED_DIR/pose_data_val.npy \
    --mapping $PROCESSED_DIR/pose_data_action_mapping.json \
    --output-dir $OUTPUT_DIR

# Step 3: Visualize some samples for verification
echo "Creating visualizations for verification..."
python data_visualization.py \
    --input $MAT_DIR \
    --output $PROCESSED_DIR/visualizations \
    --mapping $PROCESSED_DIR/pose_data_action_mapping.json \
    --num-samples 5

# Step 4: Train the Pose-LLaVA model
echo "Training Pose-LLaVA model..."
deepspeed pose_train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path $OUTPUT_DIR/pose_train.json \
    --tune_mm_projector_only True \
    --mm_projector_type mlp2x_gelu \
    --output_dir $MODEL_OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --bf16 True \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True


python -m llava.pose_train \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path $OUTPUT_DIR \
    --tune_mm_projector_only True \
    --mm_projector_type mlp2x_gelu \
    --output_dir $MODEL_OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --bf16 True \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
echo "Training complete! Model saved to $MODEL_OUTPUT_DIR"