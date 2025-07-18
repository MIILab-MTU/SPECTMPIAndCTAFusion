import os
import shutil
from pathlib import Path

base_path = r"path/to/spect"
output_base_path = r"id"

os.makedirs(output_base_path, exist_ok=True)

patient_list = []
for patient_folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, patient_folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                patient_name = os.path.splitext(file)[0]
                patient_list.append(patient_name)

for patient_name in patient_list:
    patient_output_path = os.path.join(output_base_path, patient_name)
    os.makedirs(patient_output_path, exist_ok=True)
    found_txt = False
    for patient_folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, patient_folder)
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, f"{patient_name}.csv")
            if os.path.exists(csv_file):
                for file in os.listdir(folder_path):
                    if 'ijkspect' in file.lower() and file.endswith('.txt'):
                        source_txt = os.path.join(folder_path, file)
                        dest_txt = os.path.join(patient_output_path, file)
                        shutil.copy2(source_txt, dest_txt)
                        print(f"Copied {file} to {patient_output_path}")
                        found_txt = True
                break
    if not found_txt:
        print(f"No ijkspect txt file found for patient: {patient_name}")

print("Patient List:", patient_list)