# ♻️ Garbage Classification API - Developer Documentation

Hey there! Welcome to the developer documentation for the **Garbage Classification API**. This backend is built using **FastAPI** and hosts our deep learning image classification model (a fine-tuned **EfficientNetV2-B2** architecture). 

Whether you are hooking this up to a frontend web app, a mobile app, or a smart trash bin IoT device, this guide will help you integrate it in minutes.

---

## 📍 Base URL
* **Production Deployment:** `https://shanmugaraj27-garbage-classification-backend.hf.space`
* **Local Development:** `http://localhost:8000`

---

## 🚦 Endpoints Quick Reference

| Endpoint | HTTP Method | Auth Required | Description |
| :--- | :--- | :--- | :--- |
| `GET /` | `GET` | No | Health check, server status, and model load check. |
| `GET /categories` | `GET` | No | Fetches structural waste categories and disposal guides. |
| `POST /predict` | `POST` | No | Accepts an image and returns the AI classification & recycling tips. |

---

## 📖 Endpoint Details

### 1. Health Check
Checks if the API is active and verifies if the ML model (`.keras` file) is fully loaded in memory.

* **Path:** `/`
* **Method:** `GET`
* **Headers:** `Accept: application/json`

#### 📥 Response Example (`200 OK`)
```json
{
  "message": "Garbage Classification API is running",
  "model_loaded": true
}
```

---

### 2. Get Waste Categories Metadata
Fetches our educational categories, including description, recycling viability, specific item examples, and actionable sorting tips.

* **Path:** `/categories`
* **Method:** `GET`

#### 📥 Response Example (`200 OK`)
```json
{
  "categories": [
    {
      "name": "Cardboard",
      "description": "Boxes, packaging materials",
      "icon": "📦",
      "examples": ["Shipping boxes", "Cereal boxes", "Pizza boxes"],
      "recyclable": true,
      "tips": "Remove tape and flatten boxes before recycling. Clean cardboard recycles better."
    },
    {
      "name": "Plastic",
      "description": "Bottles, bags, containers",
      "icon": "🧴",
      "examples": ["Water bottles", "Plastic bags", "Food containers"],
      "recyclable": true,
      "tips": "Check recycling number. Clean containers and remove caps. Not all plastics are recyclable."
    }
    // ...other categories like Glass, Metal, Paper, Trash
  ]
}
```

---

### 3. Classify Waste Image (AI Prediction)
This is the core engine endpoint. You send an image file, and the model classifies it into one of 6 classes: `Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, or `Trash`.

* **Path:** `/predict`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`

#### 📤 Request Body Parameters
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | `binary` (file upload) | **Yes** | The image file to analyze. Allowed formats: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`. Max size limit: `16MB`. |

#### 📥 Response Example (`200 OK`)
```json
{
  "success": true,
  "prediction": "Plastic",
  "confidence": 0.9842,
  "confidence_percentage": "98.4%",
  "category_info": {
    "name": "Plastic",
    "description": "Bottles, bags, containers",
    "icon": "🧴",
    "examples": ["Water bottles", "Plastic bags", "Food containers"],
    "recyclable": true,
    "tips": "Check recycling number. Clean containers and remove caps. Not all plastics are recyclable."
  },
  "all_predictions": [
    { "class": "Cardboard", "confidence": 0.0012, "percentage": "0.1%" },
    { "class": "Glass", "confidence": 0.0034, "percentage": "0.3%" },
    { "class": "Metal", "confidence": 0.0051, "percentage": "0.5%" },
    { "class": "Paper", "confidence": 0.0021, "percentage": "0.2%" },
    { "class": "Plastic", "confidence": 0.9842, "percentage": "98.4%" },
    { "class": "Trash", "confidence": 0.0040, "percentage": "0.4%" }
  ]
}
```

#### 🚨 Error Statuses
* `400 Bad Request`: Mismatched/invalid file format (must be PNG, JPG, JPEG, GIF, or BMP).
* `413 Payload Too Large`: Uploaded image exceeds the `16MB` limit.
* `503 Service Unavailable`: The ML model weights failed to load at server startup.

---

## 💻 Developer Code Snippets

Here's how to integrate and call the `/predict` endpoint from different programming languages:

### 1. JavaScript (Fetch API)
```javascript
const uploadImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await fetch('https://shanmugaraj27-garbage-classification-backend.hf.space/predict', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    console.log('AI Prediction:', data.prediction);
    console.log('Confidence Score:', data.confidence_percentage);
    console.log('Eco Tip:', data.category_info.tips);
  } catch (error) {
    console.error('Upload failed:', error);
  }
};
```

### 2. Python (Requests)
```python
import requests

api_url = "https://shanmugaraj27-garbage-classification-backend.hf.space/predict"
file_path = "sample_bottle.jpg"

with open(file_path, "rb") as image_file:
    files = {"file": (file_path, image_file, "image/jpeg")}
    response = requests.post(api_url, files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']} ({result['confidence_percentage']})")
    print(f"Action Tip: {result['category_info']['tips']}")
else:
    print(f"Failed with code {response.status_code}: {response.text}")
```

### 3. cURL (Terminal Quick Testing)
```bash
curl -X POST "https://shanmugaraj27-garbage-classification-backend.hf.space/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

---

## 🐳 Running locally (Docker vs Uvicorn)

If you are debugging the backend locally:

### Option A: standard python command
Make sure your virtual environment is active, then run:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Option B: Docker Container
We have bundled a robust, secure non-root `Dockerfile` in the directory. To run:
```bash
docker build -t garbage-classifier-backend .
docker run -p 7860:7860 garbage-classifier-backend
```
*Your local containerized API will now be running on `http://localhost:7860`!*
