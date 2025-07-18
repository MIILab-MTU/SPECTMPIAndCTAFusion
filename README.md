# Image fusion using 3D left heart point cloud for SPECT and CT


## Data 
- The left center surface point cloud of SPECT and CT needs to be placed in the following directory
- `file_process/id`

### Data Format
Point cloud data in. txt format

**Example Directory Structure**:
```
ROOT/file_process
├── id/ 
│   ├── data_id_01/ 
│   │   ├── ijkcta.txt 
│   │   ├── ijkspect.txt 
│   ├── data_id_02/ 
│   │   ├── ijkcta.txt 
│   │   ├── ijkspect.txt 

```

## Dependencies

Install the dependencies using the provided `requirements.txt`:
```
pip install -r requirements.txt
```
### Key Dependencies
- PyTorch
- NumPy
- matlabengineforpython
- pyvista
- openpyxl

## Usage
Update the parameters in the `main_log.py` file:

```yaml
xlsx_dir: "path/to/all_data_info.xlsx"
results_file: "path to result"
visual: "visual"
root_dir: "path to input point cloud"
cloud_dir: "path to point cloud result"
...
```
Run the main script:
```
python main_log.py
```



