import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations

def load_text_files(folder_path):
    """Loads text files from the given folder while ignoring the `unique_output` folder."""
    text_files = {}

    for root, dirs, files in os.walk(folder_path):
        # Skip the "unique_output" folder
        if "unique_output" in root:
            continue  

        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_files[file_name] = f.read()

    print(f"Final text files loaded: {list(text_files.keys())}")  # Debugging line
    return text_files


def calculate_cosine_similarity(text_files):
    """Computes cosine similarity between text files."""
    file_names = list(text_files.keys())
    corpus = list(text_files.values())

    # Convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compute cosine similarity
    cosine_similarities = (tfidf_matrix * tfidf_matrix.T).toarray()
    
    return file_names, cosine_similarities

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd

def find_redundant_texts(folder_path, threshold):
    """Find redundant text files based on cosine similarity."""
    text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    file_paths = [os.path.join(folder_path, f) for f in text_files]
    
    if not file_paths:
        return None

    contents = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            contents.append(f.read())

    # compute TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer().fit_transform(contents)
    sim_matrix = cosine_similarity(vectorizer)

    redundant_pairs = []
    for i in range(len(text_files)):
        for j in range(i+1, len(text_files)):
            sim_pct = sim_matrix[i, j] * 100
            if sim_pct >= threshold:
                # split on whitespace (you can improve tokenization if needed)
                tokens1 = set(contents[i].split())
                tokens2 = set(contents[j].split())
                common = tokens1 & tokens2

                # join overlapping tokens into a string
                highlighted = ", ".join(sorted(common)) if common else ""

                redundant_pairs.append({
                    "Text1": text_files[i],
                    "Text2": text_files[j],
                    "CosineSimilarity": f"{sim_pct:.2f}%",
                    "HighlightedRedundancy": highlighted
                })

    return pd.DataFrame(redundant_pairs)
    
