from datasets import load_dataset
from PIL import Image
import os

# Load the dataset
ds = load_dataset("benjamin-paine/imagenet-1k-32x32", split="train")

# Define the output directory
output_dir = "imagenet_32x32"
os.makedirs(output_dir, exist_ok=True)

# Save the images
for idx, item in enumerate(ds):
    image = item["image"]  # Access the image
    label = item["label"]  # Access the label

    # Create a directory for each label
    label_dir = os.path.join(output_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

    # Save the image
    file_path = os.path.join(label_dir, f"{idx}.png")
    image.save(file_path, "PNG")

    # Optional: Print progress every 100 images
    if idx % 100 == 0:
        print(f"Saved {idx} images")

print(f"Images saved in {output_dir}")
