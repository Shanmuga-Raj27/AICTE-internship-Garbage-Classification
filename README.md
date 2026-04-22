# Garbage Classification Using Transfer Learning (Modernized)

## Project Overview
This project implements a deep learning solution for garbage classification using transfer learning with EfficientNetV2B2. The system has been modernized into a split-architecture application:
- **Backend**: FastAPI with TensorFlow, deployed on Hugging Face Spaces (Docker).
- **Frontend**: Vue 3 with Tailwind CSS, deployed on Vercel/Netlify.

## Model Architecture
- **Base Model**: EfficientNetV2B2
- **Input Shape**: (224, 224, 3)
- **Pre-trained Weights**: ImageNet
- **Classes**: Cardboard, Glass, Metal, Paper, Plastic, Trash

## Project Structure
```
Garbage-Classification/
├── backend/
│   ├── Dockerfile              # Hugging Face deployment config
│   ├── main.py                # FastAPI Application
│   ├── model.py               # Model training & logic
│   ├── requirements.txt       # Backend dependencies
│   └── best_model_finetuned224.keras
├── frontend/                  # Vue 3 Application
│   ├── src/
│   │   ├── views/
│   │   │   ├── Home.vue       # Educational landing page
│   │   │   └── Classifier.vue # Image upload & AI analysis
│   └── vite.config.js
└── README.md
```

## Setup & Deployment

### Backend (Hugging Face Spaces)
The backend is Dockerized for easy deployment on Hugging Face Spaces.
1. Create a new Space on Hugging Face (Docker SDK).
2. Upload the contents of the `backend/` directory.
3. The `Dockerfile` will automatically handle setup and binding to port 7860.

### Frontend (Vercel/Netlify)
1. Deploy the `frontend/` directory to Vercel or Netlify.
2. Set the `VITE_API_URL` environment variable to your Hugging Face Space URL.
3. Build command: `npm run build`
4. Output directory: `dist`

### Local Development
1. **Backend**:
   - Install dependencies: `pip install -r backend/requirements.txt`
   - Run: `uvicorn backend.main:app --reload --port 10000`
2. **Frontend**:
   - Install dependencies: `npm install`
   - Create a `.env` file with `VITE_API_URL="http://localhost:10000"`
   - Run: `npm run dev`

## Technical Implementation
- **Backend**: FastAPI, TensorFlow, PIL, python-dotenv
- **Frontend**: Vue 3, Vite, Tailwind CSS, Lucide Icons
- **Deployment**: Docker (HF Spaces), Vercel/Netlify

## Developer
**Shanmugaraj** (Modernization by Antigravity)

---
*This project provides an efficient solution for automated garbage classification to improve recycling processes and waste management.*
