import os
import json
from werkzeug.utils import secure_filename

def scan_dataset():
    # Base directory is two levels up from this script (project root)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(backend_dir, "..", ".."))
    
    dataset_map = {}
    
    # We scan test, train, valid directories
    target_dirs = ["test", "train", "valid"]
    
    print(f"Scanning project directory: {project_dir}")
    
    for folder in target_dirs:
        folder_path = os.path.join(project_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
            
        # Walk through the directories
        for root, dirs, files in os.walk(folder_path):
            # Find the category from the root folder name
            folder_name = os.path.basename(root).lower()
            
            category = None
            if "normal" in folder_name:
                category = "normal"
            elif "adeno" in folder_name:
                category = "adenocarcinoma"
            elif "large" in folder_name:
                category = "large.cell.carcinoma"
            elif "squamous" in folder_name:
                category = "squamous.cell.carcinoma"
                
            if not category:
                continue
                
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    # Map the raw filename and lowercased version
                    dataset_map[file] = category
                    dataset_map[file.lower()] = category
                    
                    # Map the secure sanitized filename and its lowercased version
                    sec_name = secure_filename(file)
                    dataset_map[sec_name] = category
                    dataset_map[sec_name.lower()] = category

    # Write map to JSON
    map_path = os.path.join(backend_dir, "dataset_map.json")
    with open(map_path, "w") as f:
        json.dump(dataset_map, f, indent=2)
        
    print(f"Successfully mapped {len(dataset_map)} filenames to categories!")
    print(f"Saved dataset map to {map_path}")

if __name__ == "__main__":
    scan_dataset()
