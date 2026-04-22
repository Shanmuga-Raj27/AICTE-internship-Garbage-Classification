# Role
You are an Expert Python Full Stack Developer and AI Deployment Specialist. You specialize in modernizing legacy codebase architectures, specifically migrating from Flask to FastAPI, and building lightweight, modern, mobile-friendly frontends using Vue.js and Tailwind CSS. 

# Task Objective
Analyze the provided Flask deployment script for a Garbage Classification AI model. Modernize the entire stack by transitioning the backend to FastAPI and completely rewriting the frontend in Vue.js. 

**STRICT CONSTRAINT:** Do absolutely nothing to any `.ipynb` files. This migration applies strictly to the `.py` backend and the web frontend.

# Phase 1: Environment Setup & Configuration
1.  **Package Management:** Provide terminal commands to initialize the project using the `uv` package manager. 
2.  **Virtual Environment:** Show how to create and activate the `venv` using `uv`.
3.  **Dependencies:** Generate a `requirements.txt` (or `pyproject.toml`) including `fastapi`, `uvicorn`, `tensorflow`, `Pillow`, `python-dotenv`, `python-multipart`, and any other necessary libraries.
4.  **Environment Variables:** Create a `.env` file template containing:
    * `SECRET_KEY`
    * `API_KEY` (if applicable for future scaling)
    * `MODEL_PATH` (defaulting to the local `.keras` file)
    * `UPLOAD_FOLDER_PATH`

# Phase 2: Backend Migration (Flask to FastAPI)
Analyze the provided `model.py` and rewrite it into a new file named `backend/main.py`. 
* **Framework:** Use FastAPI.
* **Syntax & Architecture fixes:** * Remove all Jupyter-specific threading, `app.run()`, and `__IPYTHON__` checks. FastAPI uses Uvicorn.
    * Convert the synchronous file handling and prediction routes to `async def`.
    * Use `UploadFile` from FastAPI for image handling.
    * Implement CORS middleware to allow the new Vue frontend to communicate with the API.
    * Keep the EfficientNet preprocessing logic and the `waste_categories` dictionary exactly as they are.
    * Implement proper error handling using `HTTPException`.

# Phase 3: Frontend Modernization (Vue.js)
Analyze the implied requirements from the Flask backend logic to build a modern, mobile-friendly Vue.js frontend (Single Page Application or Vite setup). Do not use the old HTML/CSS/JS files; start fresh.
* **Page 1: Landing Page.** An educational hub displaying the `waste_categories` data (Cardboard, Glass, Metal, Paper, Plastic, Trash) with their descriptions, icons, and tips.
* **Page 2: Image Classification Model.** A clean, mobile-optimized drag-and-drop or tap-to-upload interface. It should send the image to the FastAPI backend, display a loading state, and render the prediction result, confidence percentage, and the specific recycling tips for that category.
* **Styling:** Use Tailwind CSS for a modern, lightweight, responsive UI. 

# Source Code for Analysis

### Backend Source (`model.py`):
```python
# Garbage Classification Flask Web Application
# Educational platform for waste management and AI model deployment
# Developed by Shanmugaraj

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import os
import threading
import time

# Initialize Flask application
app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Load your trained model
try:
    model = load_model("best_model_finetuned224.keras")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  Make sure 'best_model_finetuned224.keras' is in the project directory")
    model = None

# Class labels matching your model
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Waste category information for educational content
waste_categories = [
    {
        "name": "Cardboard", 
        "description": "Boxes, packaging materials", 
        "icon": "📦",
        "examples": ["Shipping boxes", "Cereal boxes", "Pizza boxes"],
        "recyclable": True,
        "tips": "Remove tape and flatten boxes before recycling. Clean cardboard recycles better."
    },
    {
        "name": "Glass", 
        "description": "Bottles, jars, containers", 
        "icon": "🍶",
        "examples": ["Wine bottles", "Mason jars", "Glass containers"],
        "recyclable": True,
        "tips": "Remove lids and rinse clean. Clear glass has the highest recycling value."
    },
    {
        "name": "Metal", 
        "description": "Cans, foil, metal objects", 
        "icon": "🥫",
        "examples": ["Aluminum cans", "Tin foil", "Metal containers"],
        "recyclable": True,
        "tips": "Rinse food residue and remove labels when possible. Aluminum recycles infinitely."
    },
    {
        "name": "Paper", 
        "description": "Newspapers, magazines, documents", 
        "icon": "📄",
        "examples": ["Newspapers", "Office paper", "Magazines"],
        "recyclable": True,
        "tips": "Keep paper dry and clean. Remove staples and plastic windows from envelopes."
    },
    {
        "name": "Plastic", 
        "description": "Bottles, bags, containers", 
        "icon": "🧴",
        "examples": ["Water bottles", "Plastic bags", "Food containers"],
        "recyclable": True,
        "tips": "Check recycling number. Clean containers and remove caps. Not all plastics are recyclable."
    },
    {
        "name": "Trash", 
        "description": "Non-recyclable waste", 
        "icon": "🗑️",
        "examples": ["Food waste", "Contaminated items", "Mixed materials"],
        "recyclable": False,
        "tips": "Items too contaminated or made of mixed materials should go to regular trash."
    }
]

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Resize image to model input size (224x224)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        
        # Ensure correct shape
        if img_array.shape != (224, 224, 3):
            return None
        
        # Apply EfficientNetV2 preprocessing
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Home page route - Educational content about garbage classification"""
    return render_template('index.html', 
                         categories=waste_categories,
                         model_available=model is not None)

@app.route('/model')
def model_page():
    """Model deployment page route - ML model interface"""
    return render_template('model.html', 
                         model_available=model is not None)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint - Process uploaded images and return classification results"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure best_model_finetuned224.keras is in the project directory.'
            })
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded. Please select an image to classify.'
            })
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected. Please choose an image file.'
            })
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files only.'
            })
        
        # Read and process image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image for model
        processed_image = preprocess_image(image)
        if processed_image is None:
            return jsonify({
                'success': False,
                'error': 'Error processing image. Please try a different image.'
            })
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(prediction[predicted_class_index])
        
        # Get category details
        category_info = next((cat for cat in waste_categories if cat['name'] == predicted_class_name), None)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': predicted_class_name,
            'confidence': confidence,
            'confidence_percentage': f"{confidence:.1%}",
            'category_info': category_info,
            'all_predictions': [
                {
                    'class': class_names[i],
                    'confidence': float(prediction[i]),
                    'percentage': f"{float(prediction[i]):.1%}"
                }
                for i in range(len(class_names))
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred during prediction: {str(e)}'
        })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Please upload an image smaller than 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle page not found error"""
    return render_template('index.html', categories=waste_categories), 404

@app.errorhandler(500)
def server_error(e):
    """Handle internal server error"""
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again.'
    }), 500

def create_directories():
    """Create required directories if they don't exist"""
    directories = ['templates', 'static', 'static/images']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def run_flask_app():
    """Run Flask application in a separate thread"""
    # Use port 5001 since 5000 is occupied
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5001, threaded=True)

def start_server():
    """Start the Flask server for Jupyter notebook"""
    print("🚀 GARBAGE CLASSIFICATION SERVER")
    print("=" * 50)
    
    # Create required directories
    create_directories()
    
    # System diagnostics
    print(f"📁 Current Directory: {os.getcwd()}")
    print(f"🧠 Model Status: {'✅ Loaded' if model is not None else '❌ Not Loaded'}")
    print(f"📄 Model File Exists: {'✅ Yes' if os.path.exists('best_model_finetuned224.keras') else '❌ No'}")
    
    print("\n🚀 Starting Flask Server on Port 5001...")
    print("⚠️  Using Port 5001 because Port 5000 is occupied")
    
    # Run Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    print("=" * 50)
    print("✅ SERVER STARTED SUCCESSFULLY!")
    print("🌐 Access your application:")
    print("   📚 Home Page: http://localhost:5001")
    print("   🤖 Model Page: http://localhost:5001/model")
    print("   🌐 Network Access: [http://192.168.1.8:5001](http://192.168.1.8:5001)")
    print("\n💡 Make sure you have the following files/folders:")
    print("   📁 templates/ (folder)")
    print("   📁 static/ (folder)")  
    print("   🎯 best_model_finetuned224.keras (model file)")
    print("\n🛑 To stop: restart notebook kernel")
    print("=" * 50)
    
    return "Server running on port 5001!"

if __name__ == '__main__':
    try:
        # Check if running in Jupyter
        __IPYTHON__
        print("📓 Jupyter environment detected")
        print("🔧 Use: start_server()")
    except NameError:
        # Running from command line
        create_directories()
        print("💻 Running from command line on port 5001")
        app.run(debug=True, host='0.0.0.0', port=5001)


# In[3]:
start_server()