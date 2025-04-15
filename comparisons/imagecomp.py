import os
from PIL import Image
import imagehash
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def get_image_files(root_dir, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif")):
    """
    Recursively collects image file paths from root_dir.
    """
    image_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(valid_exts):
                image_files.append(os.path.join(dirpath, filename))
    return image_files

def compute_phash(image_path):
    """
    Computes the perceptual hash (pHash) for an image.
    """
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def compute_ssim(image_path1, image_path2, size=(256, 256)):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    Both images are loaded in grayscale, resized to `size`, and compared.
    Returns a score between 0 and 1 (1 means identical).
    """
    try:
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return None
        # Resize both images to a common size
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)
        score, _ = ssim(img1, img2, full=True)
        return score
    except Exception as e:
        print(f"SSIM error comparing {image_path1} and {image_path2}: {e}")
        return None

def main():
    root_folder = input("Enter path_to_your_folder containing Images")  # Set this to your target folder
    image_paths = get_image_files(root_folder)
    
    # Build a list of dictionaries with file path and pHash value
    images_data = []
    for path in image_paths:
        ph = compute_phash(path)
        if ph is not None:
            images_data.append({"path": path, "phash": ph})
    
    num_images = len(images_data)
    print(f"Found {num_images} images. Comparing each unique pair...\n")
    
    # For each unique pair, compute pHash similarity and SSIM
    for i in range(num_images):
        for j in range(i+1, num_images):
            path1 = images_data[i]["path"]
            path2 = images_data[j]["path"]
            hash1 = images_data[i]["phash"]
            hash2 = images_data[j]["phash"]
            
            # pHash similarity: lower Hamming distance means higher similarity.
            # For an 8x8 pHash (64 bits), maximum distance is 64.
            hamming_distance = hash1 - hash2
            phash_similarity = ((64 - hamming_distance) / 64) * 100
            
            # Compute SSIM similarity
            ssim_score = compute_ssim(path1, path2)
            ssim_percent = ssim_score * 100 if ssim_score is not None else None
            
            print(f"Comparing:\n  {path1}\n  {path2}")
            print(f"  pHash Similarity: {phash_similarity:.2f}%")
            if ssim_percent is not None:
                print(f"  SSIM: {ssim_percent:.2f}%\n")
            else:
                print("  SSIM: N/A\n")

if __name__ == "__main__":
    main()
