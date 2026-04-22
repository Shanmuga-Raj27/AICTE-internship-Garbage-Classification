# Role
Expert Python/Vue Full Stack Developer and Cloud DevOps Specialist.

# Task Objective
Prepare this repository for a split-architecture production deployment:
1. **Backend:** FastAPI + TensorFlow deployed on **Hugging Face Spaces (Docker)**.
2. **Frontend:** Vue 3 deployed on **Vercel or Netlify**.

Please review the full codebase and generate the necessary files and updates to achieve production readiness.

# Step 1: Hugging Face Backend Setup (Docker)
1. Generate a `Dockerfile` in the `backend` directory.
2. **Requirements for HF Spaces Dockerfile:**
   - Use `python:3.11` as the base image.
   - Install dependencies from `requirements.txt` using `--no-cache-dir`.
   - **Crucial:** Set up a non-root user (`useradd -m -u 1000 user`) and switch to it. Hugging Face Spaces fail if run as root.
   - Expose port `7860` (Hugging Face default).
   - Set the CMD to run Uvicorn on `0.0.0.0:7860`.

# Step 2: Backend CORS & Security (`main.py`)
1. Update `main.py` to properly configure `CORSMiddleware`.
2. Set `allow_origins=["*"]` initially for testing, but add a clear comment on where I should paste my future Vercel/Netlify domain (e.g., `https://my-garbage-app.vercel.app`) to lock it down for production.
3. Ensure no hardcoded local paths (like `C:\Users\...`) exist. Ensure the model path is loaded via `os.getenv()` or a relative path directly in the `backend` folder.

# Step 3: Frontend API URL Configuration
1. Analyze the Vue.js frontend where the API requests (e.g., `fetch` or `axios` calls to `/predict`) are made.
2. Refactor the hardcoded `http://localhost:8000` URL to use Vite environment variables.
3. Show me how to implement `import.meta.env.VITE_API_URL` so the app knows to use localhost during development, but uses the Hugging Face Space URL in production.
4. Provide the exact `.env` file content I need to create in my `frontend` folder.

# Step 4: Final Production Checklist
Review `requirements.txt`, `vite.config.js`, and `package.json`. Ensure:
- `fastapi`, `uvicorn`, `python-multipart`, and `python-dotenv` are in `requirements.txt`.
- The frontend build command (`npm run build`) is correctly outputting to the `dist` folder for Vercel/Netlify.

# Deliverables
Output the exact code for:
1. `backend/Dockerfile`
2. The updated CORS section of `backend/main.py`
3. The updated API fetch logic for `frontend/src/views/Classifier.vue` (or relevant file)
4. The `.env` template for the frontend.