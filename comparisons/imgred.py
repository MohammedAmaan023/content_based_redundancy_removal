import os
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load pre-trained MobileNetV2 with global average pooling (feature vector shape: 1280)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def get_image_files(root_dir, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif")):
    """Recursively collect image file paths from the given root directory."""
    return [str(p) for p in Path(root_dir).rglob("*") if p.suffix.lower() in valid_exts]

def load_and_preprocess(image_path, target_size=(224, 224)):
    """Load an image, convert to RGB, resize, and apply preprocessing."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        arr = np.array(img)
        arr = preprocess_input(arr.astype(np.float32))
        return arr
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def get_variants(image, crop_ratio=0.8):
    """
    Generate a list of transformed variants for a given PIL image.
    Returns list of tuples: (variant_name, variant_image).
    Variants include: original, rotations (90, 180, 270),
    horizontal flip, vertical flip, and a central crop.
    """
    variants = [("original", image)]
    variants.append(("rotate90", image.rotate(90, expand=True)))
    variants.append(("rotate180", image.rotate(180, expand=True)))
    variants.append(("rotate270", image.rotate(270, expand=True)))
    variants.append(("flip_lr", image.transpose(Image.FLIP_LEFT_RIGHT)))
    variants.append(("flip_tb", image.transpose(Image.FLIP_TOP_BOTTOM)))
    
    w, h = image.size
    crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    variants.append(("central_crop", image.crop((left, top, left + crop_w, top + crop_h))))
    return variants

def extract_feature_from_array(arr):
    """
    Given a preprocessed image array (of shape target_size + (3,)),
    add batch dimension and extract the feature vector using MobileNetV2.
    """
    arr_batch = np.expand_dims(arr, axis=0)
    features = model.predict(arr_batch, verbose=0)
    # Normalize feature vector
    norm_features = features / np.linalg.norm(features)
    return norm_features

def get_variant_features(image, target_size=(224, 224)):
    """
    For a given PIL image, generate variants and compute their feature vectors.
    Returns a dict: {variant_name: feature_vector}
    """
    variants = get_variants(image)
    features_dict = {}
    for name, variant in variants:
        # Resize variant to the target size and preprocess
        variant_resized = variant.resize(target_size)
        arr = np.array(variant_resized)
        arr = preprocess_input(arr.astype(np.float32))
        feat = extract_feature_from_array(arr)
        features_dict[name] = feat
    return features_dict

def compare_feature_sets(features1, features2):
    """
    Given two dictionaries of features (from different variants), compute the cosine similarity
    for every pair of variants and return the maximum similarity (as a percentage) along with details.
    """
    best_similarity = 0
    best_detail = None
    for (name1, feat1), (name2, feat2) in product(features1.items(), features2.items()):
        # Compute cosine similarity (features are 2D arrays of shape (1, D))
        sim = cosine_similarity(feat1, feat2)[0][0] * 100
        if sim > best_similarity:
            best_similarity = sim
            best_detail = (name1, name2, sim)
    return best_similarity, best_detail

def main():
    root_folder = input("Enter the root folder path containing images: ").strip()
    try:
        threshold = float(input("Enter matching threshold percentage (0-100): ").strip())
    except ValueError:
        print("Invalid threshold. Using default value of 80%.")
        threshold = 80.0

    image_paths = get_image_files(root_folder)
    num_images = len(image_paths)
    if num_images < 2:
        print("Not enough images to compare.")
        return

    print(f"\nFound {num_images} images. Extracting features...")

    # Pre-compute feature dictionaries for each image (for all variants)
    images_data = []
    for path in image_paths:
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error opening {path}: {e}")
            continue
        variant_features = get_variant_features(pil_img)
        images_data.append({"path": path, "features": variant_features})
    
    print("Comparing images...\n")
    # Compare every unique pair
    for i in range(len(images_data)):
        for j in range(i + 1, len(images_data)):
            data1 = images_data[i]
            data2 = images_data[j]
            best_sim, detail = compare_feature_sets(data1["features"], data2["features"])
            if best_sim >= threshold:
                var1, var2, sim_val = detail
                print("------------------------------------------------------------")
                print(f"Image 1: {data1['path']}")
                print(f"Image 2: {data2['path']}")
                print(f"Best variant match: {var1} vs {var2}")
                print(f"Cosine Similarity: {sim_val:.2f}%\n")

if __name__ == "__main__":
    main()
