from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from .imgred import load_and_preprocess, extract_feature_from_array, compare_feature_sets
import os,shutil, re, glob,base64
from pathlib import Path
from datetime import datetime
from .utils import compare_images
from .text_similarity import find_redundant_texts 
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import tempfile
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity


def home(request):
    
    return render(request, 'comparisons/home.html')

def process_text(request):
    if request.method == 'POST':
        folder_path = request.POST.get('folder_path')
        threshold = request.POST.get('threshold')

        if not folder_path or not os.path.exists(folder_path):
            return render(request, 'comparisons/text.html', {"error": "Please enter a valid folder path."})

        try:
            threshold = float(threshold)
        except ValueError:
            return render(request, 'comparisons/text.html', {"error": "Invalid threshold value. Please enter a number."})

        # Run redundancy detection with cosine similarity
        result_df = find_redundant_texts(folder_path, threshold)

        if result_df is not None and not result_df.empty:
            # Keep column names consistent with previous outputs
            result_df = result_df.rename(columns={
                "Text 1": "Text1",
                "Text 2": "Text2",
                "Cosine Similarity": "CosineSimilarity"
            })
            
            # Convert DataFrame to list of dictionaries for template rendering
            result = result_df.to_dict('records')
            return render(request, 'comparisons/text_results.html', {
                'result': result,
                'folder_path': folder_path,
                'threshold': threshold
            })
        else:
            return render(request, 'comparisons/text.html', {
                "message": "No redundant texts found with the given threshold."
            })
    
    return render(request, 'comparisons/text.html')



    
def text_redundancy_view(request):
    
    result_df = None


    if request.method == "POST":
        folder_path = request.POST.get("folder_path")
        threshold = request.POST.get("threshold")
        folder_path_value = folder_path  # Store entered value
        threshold_value = threshold

        if not folder_path or not threshold:
            return render(request, "comparisons/text.html", {
                "error": "Please enter a valid folder path and threshold value.",
                "folder_path_value": folder_path_value,
                "threshold_value": threshold_value
            })

        try:
            threshold = float(threshold)
            print(f"Threshold Value: {threshold}")
        except ValueError:
            return render(request, "comparisons/text.html", {
                "error": "Invalid threshold value. Please enter a number.",
                "folder_path_value": folder_path_value,
                "threshold_value": threshold_value
            })


        # Run text similarity check
        print(f"Processing Folder: {folder_path}")
        result_df = find_redundant_texts(folder_path, threshold)

        if result_df is not None and not result_df.empty:
             
            result_df = result_df.rename(columns={
                "Text 1": "Text1",
                "Text 2": "Text2",
                "Cosine Similarity": "CosineSimilarity"
            })
        return render(request, "comparisons/text.html", {
            "result": result_df.to_dict(orient="records") if result_df is not None else None,
            "folder_path_value": folder_path_value,
            "threshold_value": threshold_value
        })   

    return render(request, "comparisons/text.html", {
        "folder_path_value": folder_path_value,
        "threshold_value": threshold_value
    })

def text_similarity_view(request):
    if request.method == "POST":
        folder_path = request.POST.get("folder_path")
        threshold = request.POST.get("threshold")  # Get threshold input

        try:
            threshold = float(threshold)  # Convert input to float
        except ValueError:
            threshold = 80  # Default if invalid input

        result_df = find_redundant_texts(folder_path, threshold)
        return render(request, "comparisons/text.html", {"result": result_df.to_dict(orient="records")})

    
    return render(request, "comparisons/text.html")

# comparisons/views.py

from django.shortcuts import render

def text_results(request):
    # You can render a template or return something simple to test
    return render(request, 'comparisons/text_results.html')  # Replace with actual template


# ResNet for image comparison
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet_model = resnet50(pretrained=True).to(device)
resnet_model.eval()
feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_features_from_path(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(img_tensor).squeeze().cpu().numpy()
        return {
            "path": img_path,
            "features": features.flatten()
        }
    except Exception as e:
        print(f"❌ Error processing image {img_path}: {e}")
        return None



def extract_features_parallel(image_paths):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_features_from_path, path) for path in image_paths]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    return results


def compare_feature_vectors(vec1, vec2):
   sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
   return sim


def compare_images(request):
    folder_path_value = ""
    threshold_value = ""

    if request.method == 'POST':
        folder_path_value = request.POST.get('folder_path', "").strip()
        threshold_value = request.POST.get('threshold', "").strip()

        if not folder_path_value or not os.path.exists(folder_path_value):
            return render(request, 'comparisons/upload.html', {
                "error": "Invalid folder path.",
                "folder_path_value": folder_path_value,
                "threshold_value": threshold_value
            })

        try:
            threshold = float(threshold_value) if threshold_value else 90
        except ValueError:
            return render(request, 'comparisons/upload.html', {
                "error": "Invalid threshold value. Please enter a number.",
                "folder_path_value": folder_path_value,
                "threshold_value": threshold_value
            })

        image_files = glob.glob(os.path.join(folder_path_value, '*.jpg')) + \
                      glob.glob(os.path.join(folder_path_value, '*.jpeg')) + \
                      glob.glob(os.path.join(folder_path_value, '*.png'))

        if len(image_files) < 2:
            return render(request, 'comparisons/upload.html', {
                "error": "At least two images are required.",
                "folder_path_value": folder_path_value,
                "threshold_value": threshold_value
            })

        data = extract_features_parallel(image_files)
        results = []

        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                sim = compare_feature_vectors(data[i]['features'], data[j]['features'])
                if sim * 100 >= threshold:
                    results.append({
                        'image1': os.path.basename(data[i]['path']),
                        'image2': os.path.basename(data[j]['path']),
                        'similarity': f"{sim * 100:.2f}%"
                    })

        return render(request, 'comparisons/upload.html', {
            "results": results,
            "folder_path_value": folder_path_value,
            "threshold_value": threshold_value
        })

    return render(request, 'comparisons/upload.html', {
        "folder_path_value": folder_path_value,
        "threshold_value": threshold_value
    })

def compare_two_images(request):
    context = {}
    
    if request.method == 'POST':
        image1 = request.FILES.get('image1')
        image2 = request.FILES.get('image2')

        if not image1 or not image2:
            context['error'] = 'Please upload both images'
            return render(request, 'comparisons/upload.html', context)

        # Save uploaded files temporarily
        fs = FileSystemStorage()
        img1_path = fs.save(f"image1_{image1.name}", image1)
        img2_path = fs.save(f"image2_{image2.name}", image2)

        # Get full paths
        img1_full_path = fs.path(img1_path)
        img2_full_path = fs.path(img2_path)

        try:
            # Calculate cosine similarity
            img1 = cv2.imread(img1_full_path)
            img2 = cv2.imread(img2_full_path)
            
            if img1 is None or img2 is None:
                raise Exception("Error reading uploaded images")

            # Convert to grayscale
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Resize to same shape if needed
            img1_gray = cv2.resize(img1_gray, (256, 256))
            img2_gray = cv2.resize(img2_gray, (256, 256))

            # Flatten images to vectors
            features1 = img1_gray.flatten().reshape(1, -1)
            features2 = img2_gray.flatten().reshape(1, -1)

            # Calculate cosine similarity
            cosine_sim = cosine_similarity(features1, features2)[0][0] * 100

            # Get URLs for the images
            img1_url = fs.url(img1_path)
            img2_url = fs.url(img2_path)

            # Convert images to base64 for display
            _, img1_encoded = cv2.imencode('.png', img1)
            _, img2_encoded = cv2.imencode('.png', img2)
            img1_base64 = base64.b64encode(img1_encoded.tobytes()).decode('utf-8')
            img2_base64 = base64.b64encode(img2_encoded.tobytes()).decode('utf-8')

            context.update({
                'similarity': f"{cosine_sim:.2f}%",
                'img1_url': img1_url,
                'img2_url': img2_url,
                'img1_data': f"data:image/png;base64,{img1_base64}",
                'img2_data': f"data:image/png;base64,{img2_base64}"
            })

        except Exception as e:
            context['error'] = f"Error processing images: {str(e)}"
        finally:
            # Clean up temporary files
            fs.delete(img1_path)
            fs.delete(img2_path)

    return render(request, 'comparisons/upload.html', context)

def examples(request):
    return render(request, 'comparisons/example.html')
