import os
import shutil

# Original dataset directory
source_dir = "/work/debiasing/gcapitani/data/sun397/SUN397"

# Output directory for splits
output_dir = "/work/debiasing/datasets/sun397"
train_output_dir = os.path.join(output_dir, "train")
test_output_dir = os.path.join(output_dir, "val")

# Split files
test_file_path = "Testing_01.txt"
train_file_path = "Training_01.txt"

# Read split files
with open(test_file_path, "r") as f:
    test_files = f.read().strip().splitlines()

with open(train_file_path, "r") as f:
    train_files = f.read().strip().splitlines()

# Function to copy files into the output directories
def copy_files(file_list, source_dir, output_dir):
    for file_path in file_list:
        # Full path to the image in the original directory
        full_file_path = os.path.join(source_dir, file_path[1:])  # Remove the first slash

        # Destination path (keeps class structure)
        class_name = file_path.split('/')[2]  # Extract class name
        destination_dir = os.path.join(output_dir, class_name)
        os.makedirs(destination_dir, exist_ok=True)
        
        # Copy file
        try:
            shutil.copy(full_file_path, destination_dir)
        except FileNotFoundError:
            print(f"File not found: {full_file_path}")

# Copy files for train and test
copy_files(train_files, source_dir, train_output_dir)
copy_files(test_files, source_dir, test_output_dir)

print(f"Splits created successfully:\n - Train: {train_output_dir}\n - Test: {test_output_dir}")
