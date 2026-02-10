#!/bin/bash
# Script to setup CelebA data from a local zip file
# Assumes the zip is at /home/dgxuser10/cryptonym/data/celeba.zip

DATA_DIR="/home/dgxuser10/cryptonym/data"
TARGET_DIR="$DATA_DIR/celeba"

echo "Setting up CelebA in $TARGET_DIR..."

if [ ! -f "$DATA_DIR/celeba.zip" ]; then
    echo "Error: $DATA_DIR/celeba.zip not found!"
    echo "Please ensure the zip file is in the data directory."
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Check if already extracted
if [ -f "$TARGET_DIR/list_attr_celeba.txt" ]; then
    echo "Files appear to be already extracted in $TARGET_DIR."
else
    echo "Unzipping celeba.zip..."
    unzip -q "$DATA_DIR/celeba.zip" -d "$DATA_DIR"
    
    # Check if unzip created a 'celeba' folder or dumped contents
    if [ -d "$DATA_DIR/celeba/celeba" ]; then
        # Nested folder case (common in some zips)
        echo "Detected nested folder, moving files..."
        mv "$DATA_DIR/celeba/celeba/"* "$TARGET_DIR/"
        rmdir "$DATA_DIR/celeba/celeba"
    elif [ -f "$DATA_DIR/img_align_celeba.zip" ]; then
        # Zip contained raw files at root, move them
        mv "$DATA_DIR/img_align_celeba.zip" "$TARGET_DIR/"
        mv "$DATA_DIR/list_"* "$TARGET_DIR/"
        mv "$DATA_DIR/identity_"* "$TARGET_DIR/"
    fi
fi

# Torchvision expects 'img_align_celeba.zip' inside the root
# If the zip contained the FOLDER 'img_align_celeba', we might need to zip it back up 
# OR just ensure torchvision sees it.
# Actually torchvision checks for the folder 'img_align_celeba' first.

if [ -d "$TARGET_DIR/img_align_celeba" ]; then
    echo "Images folder found."
elif [ -f "$TARGET_DIR/img_align_celeba.zip" ]; then
    echo "Images zip found (Torchvision will unzip this)."
else
    echo "Warning: img_align_celeba not found. Check the zip structure."
fi

echo "Done. You can now run the training script."
