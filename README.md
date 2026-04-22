# Garbage Classification Using Transfer Learning (Modernized)

### 🌍 Vision: Towards a Green India
India generates over 62 million tonnes of waste annually, but only about 20% is treated. Mismanagement of waste is a significant hurdle to the **Swachh Bharat (Clean India) Mission**. This project leverages Artificial Intelligence to automate the identification and classification of waste, empowering citizens and industries to sort garbage at the source—the first and most crucial step toward a sustainable, circular economy.

🔗 **[Live Demo: Garbage Classification Platform](https://garbage-classification-frontend.netlify.app/)**

---

## 🚀 Project Overview
This is a full-stack, AI-powered web application that identifies various types of waste in real-time. By utilizing **Transfer Learning**, the system achieves high accuracy with minimal computational overhead, making it viable for real-world environmental monitoring and educational awareness.

### 🏗️ Split-Architecture Design
To ensure high availability and professional-grade performance, the project follows a decoupled architecture:
- **Scalable AI Backend**: A containerized FastAPI service hosted on **Hugging Face Spaces**, optimized with 16GB RAM to handle heavy Deep Learning workloads.
- **Modern Responsive Frontend**: A sleek, mobile-first Vue 3 application hosted on **Netlify**, ensuring global delivery via CDN.

---

## 🛠️ Tech Stack

### **Frontend**
![Vue.js](https://img.shields.io/badge/vuejs-%2335495e.svg?style=for-the-badge&logo=vuedotjs&logoColor=%234FC08D)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)

### **Backend & AI**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

### **Deployment & DevOps**
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)
![Netlify](https://img.shields.io/badge/netlify-%23000000.svg?style=for-the-badge&logo=netlify&logoColor=#00C7B7)

---

## 🧠 Model Architecture
The core intelligence of this system is based on **EfficientNetV2B2**, a state-of-the-art convolutional neural network optimized for accuracy and parameter efficiency.

- **Transfer Learning**: Pre-trained weights from **ImageNet** were utilized to leverage high-level feature extraction.
- **Fine-Tuning**: The top layers were specialized for waste-specific features.
- **Classes**: `Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, `Trash`.

---

## 📂 Project Structure
```text
Garbage-Classification/
├── backend/                   # AI Service Engine (Python/FastAPI)
│   ├── Dockerfile             # Containerization for HF Spaces
│   ├── main.py                # FastAPI entry point & CORS config
│   ├── model.py               # ML Logic & Inference pipeline
│   ├── requirements.txt       # Production dependencies
│   └── best_model_finetuned224.keras
├── frontend/                  # User Interface (Vue 3/Vite)
│   ├── src/
│   │   ├── views/
│   │   │   ├── Home.vue       # UX-optimized Landing Page
│   │   │   └── Classifier.vue # Real-time AI Analysis View
│   ├── .env                   # Environment-driven API config
│   └── vite.config.js
└── README.md
```
## ⚙️ Deployment Strategy

### **Cloud Infrastructure (Production)**
1.  **Backend (Hugging Face)**: Deployed using a custom **Docker** container. This bypasses standard RAM limits, allowing the TensorFlow model to run flawlessly on 16GB infrastructure.
2.  **Frontend (Netlify)**: Automated CI/CD pipeline. The frontend communicates with the AI backend via the `VITE_API_URL` environment variable.

### **Local Setup**
1.  **Backend**:
    ```bash
    pip install -r backend/requirements.txt
    uvicorn backend.main:app --reload --port 10000
    ```
2.  **Frontend**:
    ```bash
    npm install
    npm run dev
    ```

---

## 💡 Key Engineering Features
- **Mobile-First UX**: Designed for the "Thumb Zone," making it easy for users to snap and classify waste on the go.
- **Environment Driven**: Fully decoupled frontend/backend connectivity via `.env` variables for seamless transition between dev and prod.
- **Eco-Theming**: Implements a "Glassmorphism" UI with support for **Light and Dark modes**, emphasizing sustainability through modern design.
- **Resource Optimized**: Leveraging Transfer Learning ensures fast inference times (~1.5s) even on CPU-only cloud servers.

---

## 👨‍💻 Developer
**Shanmugaraj** *Aspiring Python Full Stack Developer | AI Enthusiast*

> *“The greatest threat to our planet is the belief that someone else will save it.”* – Let's build a Green India together.

🔗 **[LinkedIn](https://www.linkedin.com/in/shanmugaraj27)**
🔗 **[GitHub](https://github.com/Shanmuga-Raj27)**


