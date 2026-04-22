import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="Garbage Classification API",
    description="Educational platform for waste management and AI model deployment",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace ["*"] with your frontend URL (e.g., ["https://your-frontend.onrender.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load variables
MODEL_PATH = os.getenv("MODEL_PATH", "best_model_finetuned224.keras")

# Allowable content types mapped from allowed extensions
ALLOWED_CONTENT_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp']

# Load your trained model
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Class labels matching the model
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

def preprocess_image(image: Image.Image) -> np.ndarray | None:
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

@app.get("/")
async def root():
    return {"message": "Garbage Classification API is running", "model_loaded": model is not None}

@app.get("/categories")
async def get_categories():
    """Return all waste categories info"""
    return {"categories": waste_categories}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint - Process uploaded images and return classification results"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure the model file is configured properly.")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded. Please select an image to classify.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files only.")

    try:
        # Read and process image
        image_data = await file.read()
        
        # Optional file size limit check (e.g., 16 MB)
        if len(image_data) > 16 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Please upload an image smaller than 16MB.")

        image = Image.open(io.BytesIO(image_data))

        # Preprocess image for model
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise HTTPException(status_code=400, detail="Error processing image. Please try a different image.")
        
        # Make prediction
        prediction_scores = model.predict(processed_image, verbose=0)[0]
        predicted_class_index = int(np.argmax(prediction_scores))
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(prediction_scores[predicted_class_index])
        
        # Get category details
        category_info = next((cat for cat in waste_categories if cat['name'] == predicted_class_name), None)
        
        # Prepare response
        return {
            'success': True,
            'prediction': predicted_class_name,
            'confidence': confidence,
            'confidence_percentage': f"{confidence:.1%}",
            'category_info': category_info,
            'all_predictions': [
                {
                    'class': class_names[i],
                    'confidence': float(prediction_scores[i]),
                    'percentage': f"{float(prediction_scores[i]):.1%}"
                }
                for i in range(len(class_names))
            ]
        }
        
    except HTTPException as he:
        # Re-raise HTTPExceptions so FastAPI handles them properly
        raise he
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 to be accessible on Render
    # Use the PORT environment variable if available (Render provides this)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
