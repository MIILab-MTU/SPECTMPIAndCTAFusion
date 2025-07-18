import os
import shutil
from pathlib import Path
import hashlib

spect_base_path = r"path/to/spect"
cta_base_path = r"path/to/cta"
output_base_path = r"id"

def get_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

patient_list = []
for patient_folder in os.listdir(spect_base_path):
    folder_path = os.path.join(spect_base_path, patient_folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                patient_name = os.path.splitext(file)[0]
                patient_list.append(patient_name)

for patient_name in patient_list:
    patient_output_path = os.path.join(output_base_path, patient_name)
    os.makedirs(patient_output_path, exist_ok=True)
    folder_path = os.path.join(cta_base_path, patient_name)
    found_txt = False
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if ('ijkcta' in file.lower() and
                    file.lower() != 'ijkcta1.txt' and
                    file.endswith('.txt')):
                source_txt = os.path.join(folder_path, file)
                dest_txt = os.path.join(patient_output_path, file)
                file_hash = get_file_hash(source_txt)
                shutil.copy2(source_txt, dest_txt)
                print(f"Copied {file} from {folder_path} to {patient_output_path} (Hash: {file_hash})")
                found_txt = True
    else:
        print(f"No folder found in cta/final for patient: {patient_name}")
        continue
    if not found_txt:
        print(f"No ijkcta txt file (excluding ijkcta1.txt) found for patient: {patient_name}")

print("Patient List:", patient_list)