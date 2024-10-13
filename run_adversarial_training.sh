#!/bin/bash

# Select the model for training: blip, llava, llava_next, llava_ov, qwen
MODEL_NAME="llava"

# Set the training script based on the selected model
SCRIPT_NAME="adversarial_training/train_${MODEL_NAME}.py"

# Input directory containing images to be processed
INPUT_DIR="path/to/input/images"
# Output directory to save processed images
OUTPUT_DIR="path/to/output/results"
# Directory to save logs
LOG_DIR="path/to/logs"
# Maximum perturbation epsilon for adversarial attacks
EPSILON=8/255
# Step size for the attack
ALPHA=0.5/255
# Number of steps for adversarial attack
NUM_STEPS=500
# Method to use for adversarial attack (i-fgsm or pgd)
METHOD="i-fgsm"
# Type of training (non_h for non-hallucinated, h for hallucinated images)
HALLUCINATION_TYPE="non_h"
# Whether to use category-based processing for images (true or false)
USE_CATEGORIES=true

# Run the training script
python "$SCRIPT_NAME" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --epsilon "$EPSILON" \
  --alpha "$ALPHA" \
  --num_steps "$NUM_STEPS" \
  --method "$METHOD" \
  --hallucination_type "$HALLUCINATION_TYPE" \
  $( [ "$USE_CATEGORIES" == true ] && echo "--use_categories" )