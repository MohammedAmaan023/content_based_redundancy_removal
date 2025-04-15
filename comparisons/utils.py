from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2

def compare_images(img1_path, img2_path):
    try:
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return "error", 0.0
        
        # Convert to grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Resize to same shape if needed
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

        # Flatten images to vectors
        features1 = img1.flatten().reshape(1, -1)
        features2 = img2.flatten().reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(features1, features2)[0][0] * 100

        # Determine best match type
        if np.array_equal(img1, img2):
            best_match = "original"
        elif np.array_equal(np.rot90(img1, 2), img2):
            best_match = "rotate180"
        elif np.array_equal(cv2.flip(img1, 1), img2):
            best_match = "flip_lr"
        elif np.array_equal(cv2.flip(img1, 0), img2):
            best_match = "flip_tb"
        else:
            best_match = "original"

        return best_match, similarity
    except Exception as e:
        print(f"Error comparing images: {e}")
        return "error", 0.0
