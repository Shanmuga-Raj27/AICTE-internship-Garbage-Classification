# ♻️ AI-Powered Garbage Classifier & Waste Sorting Platform

Welcome to the **Garbage Classification & Waste Sorting Platform**! This is a modernized, full-stack, decoupled machine learning web application. The platform utilizes a state-of-the-art **Transfer Learning model with EfficientNetV2-B2** to automatically classify trash items from photos and provide actionable recycling tips to help keep our planet clean.

This initiative is inspired by the **Swachh Bharat Mission**—towards a clean, green, and self-sustaining India.

🔗 **[Live Demo: Garbage Classification Platform](https://garbage-classification-frontend.netlify.app/)**

---

## 🛠️ Tech Stack

We built this platform using a modern, scalable, and decoupled architecture. Here are the core technologies powering the application:

### Frontend (User Interface & Experience)

| Technology | Badge / Logo | Purpose |
| :--- | :--- | :--- |
| **React 19** | ![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB) | Provides an ultra-responsive Single Page Application (SPA) structure. |
| **TypeScript** | ![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat-square&logo=typescript&logoColor=white) | Delivers robust type-safety and reduces development bugs. |
| **Tailwind CSS** | ![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white) | Used to craft premium glassmorphism, responsive styles, and modern dark-mode layouts. |
| **Vite** | ![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white) | Serves as our lightning-fast frontend build tool and bundler. |
| **Netlify** | ![Netlify](https://img.shields.io/badge/Netlify-00C7B7?style=flat-square&logo=netlify&logoColor=white) | Hosts the frontend with automatic deployments, HTTPS, and redirect handling. |

### Backend & Machine Learning (AI Core)

| Technology | Badge / Logo | Purpose |
| :--- | :--- | :--- |
| **FastAPI** | ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white) | Extremely fast, high-performance async framework for API routes. |
| **Python** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Powerhouse language for our machine learning and backend pipeline. |
| **TensorFlow** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | Core framework for running neural network layers and inferences. |
| **Keras** | ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) | High-level API to build, train, and manage our deep transfer learning architecture. |
| **Docker** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | Containerizes the FastAPI backend to ensure a uniform running environment everywhere. |
| **Hugging Face** | ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FFD21E?style=flat-square) | Hosts the backend server as a secure Docker Space, providing public API endpoints. |

---

## 📁 Project Structure

This project uses a clean, decoupled monorepo structure. The frontend and backend are completely isolated and self-contained, allowing for separate, hassle-free development and hosting.

```text
├── backend/
│   ├── .gitattributes                # Git LFS configuration tracking model files (.keras)
│   ├── .gitignore                    # Backend-specific environment and runtime ignores
│   ├── Dockerfile                    # Container configuration for Hugging Face Spaces
│   ├── README.md                     # Backend API metadata guide
│   ├── main.py                       # FastAPI application, routes, and image pre-processing
│   ├── model.py                      # Deep learning model definition & two-phase training logic
│   ├── requirements.txt              # Backend library dependencies (TensorFlow, Pillow, FastAPI)
│   ├── best_model224.keras           # Phase 1 model weights (frozen feature extractor base)
│   └── best_model_finetuned224.keras # Phase 2 model weights (deep fine-tuned architecture)
├── frontend/
│   ├── public/                       # Static public assets (icons, images)
│   ├── src/
│   │   ├── components/               # Layout components (Navbar, Footer, Waste Categories)
│   │   ├── views/                    # Pages & dashboards (Home Workspace, AI Sandbox)
│   │   ├── App.tsx                   # Central router & main entry point
│   │   └── main.tsx                  # React application mount script
│   ├── package.json                  # Node dependencies and npm scripts
│   └── vite.config.ts                # Vite bundler configuration
├── Dockerfile                        # Root container fallback script
├── netlify.toml                      # Netlify configuration & SPA rewrite rules for routing
└── README.md                         # Main repository workspace documentation (this file)
```

---

## 💻 Local Development & Setup

Get the entire application up and running on your local machine in just a few steps.

### 📋 Prerequisites
Make sure you have the following installed on your machine:
* **Python 3.10+**
* **Node.js (v18+)** & **npm**

---

### 🐍 1. Backend Local Setup (FastAPI & ML API)

1. Open your terminal and navigate to the `backend` folder:
   ```bash
   cd backend
   ```

2. Create a virtual environment to manage dependencies securely:
   ```bash
   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the FastAPI server locally using Uvicorn:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   * The API server will start running on **`http://localhost:8000`**.
   * Open **`http://localhost:8000/docs`** in your browser to interact with the API endpoints using the built-in Swagger UI!

---

### ⚛️ 2. Frontend Local Setup (React & Vite)

1. Open a new terminal and navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```

2. Install all node modules and package dependencies:
   ```bash
   npm install
   ```

3. Create a local environment variables file named `.env` in the root of the `frontend/` folder:
   ```env
   VITE_API_URL=http://localhost:8000
   ```
   *(This points the React application to your locally running FastAPI server instead of production)*

4. Boot up the Vite developer environment:
   ```bash
   npm run dev
   ```
   * The frontend client will run on **`http://localhost:5173`**.
   * Open your web browser and navigate there to test the application locally!

---

## 🐳 Deployment & CI/CD Pipelines

Our decoupled architecture allows both components to be hosted independently on systems optimized for their individual environments.

### 🐍 Backend Deployment (Hugging Face Spaces)
* The backend is hosted as a containerized Docker service inside **Hugging Face Spaces**.
* Hugging Face automatically detects [backend/Dockerfile](backend/Dockerfile) and provisions the server on port `7860`.
* We use a **Git Subtree** strategy to isolate and push *only* the `backend` folder history, which keeps our Hugging Face repository lightweight and free of frontend assets:
  ```bash
  # Split the backend directory into a local branch
  git subtree split --prefix=backend -b hf-split

  # Force push it to Hugging Face Spaces remote
  git push -f hf hf-split:main
  ```

### ⚛️ Frontend Deployment (Netlify)
* The React client is hosted on **Netlify**, featuring automatic deployments triggered by updates to our GitHub main branch.
* Production builds are managed by [netlify.toml](netlify.toml), setting the base directory to `frontend`, running `npm run build`, and serving from `frontend/dist`.
* Global rewrite rules are active (`/* -> /index.html 200`) to guarantee that refreshing the browser on custom React routes (like `/classify` or `/docs`) never triggers a 404 error.

---

## 👤 Developer Profile

### **Shanmugaraj R**
* **Role:** Aspiring Python backend Developer | AI Enthusiast
* **Focus:** Crafting reliable web architectures, building and deploying containerized machine learning microservices, and design-minded front-end systems with rich, user-centric interfaces.

---

## 🤝 Connect With Us

Let's collaborate on green tech, AI innovations, or clean-energy software projects! Feel free to reach out, review the source code, or check out my work:

* 💻 **GitHub Repository:** [Shanmuga-Raj27](https://github.com/Shanmuga-Raj27/AICTE-internship-Garbage-Classification.git)
* 💼 **LinkedIn Profile:** [Shanmugaraj R](https://www.linkedin.com/in/shanmugaraj27)

*Helping clean the planet, one pixel at a time.* 🌍
