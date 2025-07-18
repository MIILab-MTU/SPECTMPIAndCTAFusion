import os
import hashlib
from pathlib import Path
from collections import defaultdict

base_path = r"id"

if not os.path.exists(base_path):
    print(f"Directory {base_path} does not exist!")
    exit(1)

def get_file_hash(file_path):
    try:
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

file_hashes = defaultdict(list)
for patient_folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, patient_folder)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            file_hash = get_file_hash(file_path)
            if file_hash:
                file_hashes[file_hash].append(file_path)

found_duplicates = False
for file_hash, file_paths in file_hashes.items():
    if len(file_paths) > 1:
        found_duplicates = True
        print(f"\nFiles with identical content (Hash: {file_hash}):")
        for path in file_paths:
            print(f"  {path}")

if not found_duplicates:
    print("No files with identical content found.")

print("\nCheck complete.")