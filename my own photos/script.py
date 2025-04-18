import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    img_resized = img.resize(size, Image.Resampling.LANCZOS)  # <- Updated line
                    img_resized.save(output_path)
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

# Example usage
input_dir = "photo"
size = 224
output_dir = "photo_resized_" + str(size)
resize_images_in_folder(input_dir, output_dir, size=(size, size))
