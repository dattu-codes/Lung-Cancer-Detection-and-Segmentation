import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from inference import analyze_lung_scan

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Base directory for finding clinical sample scans
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_DIR = os.path.join(PROJECT_DIR, "test")


@app.route('/')
def index():
    """Serve the clinical dashboard home page."""
    return app.send_static_file('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Accepts an uploaded image file or a preloaded sample path.
    Runs the inference engine and returns classification & segmentation data.
    """
    try:
        sensitivity = float(request.form.get('sensitivity', 1.0))
        sample_path = request.form.get('sample_path', '')
        
        target_path = None
        
        # Scenario A: Analyze a preloaded sample path from the repo
        if sample_path:
            # Prevent directory traversal attacks
            safe_path = os.path.normpath(sample_path)
            if safe_path.startswith("..") or os.path.isabs(safe_path):
                return jsonify({"error": "Unauthorized path traversal detected"}), 403
                
            full_path = os.path.join(PROJECT_DIR, safe_path)
            if os.path.exists(full_path):
                target_path = full_path
            else:
                return jsonify({"error": f"Sample file not found at {sample_path}"}), 404
                
        # Scenario B: Analyze a newly uploaded file
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
                
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            target_path = temp_path
            
        else:
            return jsonify({"error": "No image data or sample path provided"}), 400
            
        # Run inference
        results = analyze_lung_scan(target_path, sensitivity)
        
        # Clean up temporary uploaded file if we created one
        if not sample_path and target_path and os.path.exists(target_path):
            try:
                # Give a brief moment or clean up later if occupied, but standard os.remove works
                # We can also keep it, but deleting avoids bloating the folder
                os.remove(target_path)
            except Exception as e:
                print(f"Temporary file cleanup failed: {e}")
                
        # Convert binary numpy mask to Base64 PNG image for frontend canvas rendering
        mask_np = results["mask"]
        _, buffer = cv2.imencode('.png', mask_np)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Build JSON response
        response = {
            "success": True,
            "engine": results["engine"],
            "label": results["label"],
            "confidence": results["confidence"],
            "confidences": results["confidences"],
            "mask_base64": mask_base64,
            "stats": results["stats"],
            "insights": results["insights"]
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """
    Scans the local repository's 'test' folder and discovers real clinical scans
    across adenocarcinoma, squamous cell carcinoma, large cell, and normal cases.
    """
    samples = []
    if not os.path.exists(TEST_DIR):
        # Fallback empty list if dataset test folder is missing
        return jsonify(samples)
        
    # Supported categories
    categories = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
    
    for cat in categories:
        cat_dir = os.path.join(TEST_DIR, cat)
        if not os.path.exists(cat_dir):
            continue
            
        # Gather up to 3 PNG files from each category for the dashboard grid
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        # Sort files to be deterministic
        files.sort()
        
        # Select 3 samples spaced across the list
        selected_files = []
        if len(files) > 0:
            indices = np.linspace(0, len(files) - 1, min(3, len(files)), dtype=int)
            for idx in indices:
                filename = files[idx]
                rel_path = os.path.relpath(os.path.join(cat_dir, filename), PROJECT_DIR)
                # Replace Windows backslashes with forward slashes for URL safety
                rel_path = rel_path.replace("\\", "/")
                
                selected_files.append({
                    "name": filename,
                    "category": cat.replace(".", " ").title(),
                    "path": rel_path
                })
        samples.extend(selected_files)
        
    return jsonify(samples)


@app.route('/api/sample-image', methods=['GET'])
def get_sample_image():
    """Streams a requested sample file back to the browser safely."""
    path = request.args.get('path', '')
    if not path:
        return "Missing path parameter", 400
        
    # Prevent directory traversal
    safe_path = os.path.normpath(path)
    if safe_path.startswith("..") or os.path.isabs(safe_path):
        return "Access denied", 403
        
    full_path = os.path.join(PROJECT_DIR, safe_path)
    if os.path.exists(full_path):
        return send_file(full_path)
    else:
        return f"File not found at {path}", 404


if __name__ == '__main__':
    print("Starting PulmoScan AI backend server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
