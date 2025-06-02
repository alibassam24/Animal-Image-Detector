import os
import shutil
import random

# Get base directory of the project
base_dir = os.path.join(os.getcwd(), 'data')

# Paths to validation and test directories
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Ensure the test directory exists
os.makedirs(test_dir, exist_ok=True)

# Automatically detect class names based on subdirectories in val/
class_names = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

# Loop through each class
for class_name in class_names:
    class_val_dir = os.path.join(val_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)

    # Create class folder in test directory
    os.makedirs(class_test_dir, exist_ok=True)

    # Get all images
    image_files = os.listdir(class_val_dir)
    
    # Skip if no images
    if len(image_files) == 0:
        print(f"No images found in {class_name}, skipping...")
        continue

    # Select 15% of images randomly
    num_images_to_move = max(1, int(len(image_files) * 0.15))  # At least 1 image
    images_to_move = random.sample(image_files, num_images_to_move)

    # Move the images
    for image in images_to_move:
        src = os.path.join(class_val_dir, image)
        dst = os.path.join(class_test_dir, image)
        shutil.move(src, dst)
        print(f"Moved {image} from {class_name} val to test folder.")

print(f"\n✅ Test data successfully moved to: {test_dir}")
