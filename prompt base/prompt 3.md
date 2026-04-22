# Role
Expert Python/Vue Full Stack Developer and DevOps Specialist.

# Task Objective
Prepare this repository for production deployment on Render.com. Sanitize the codebase by removing all hardcoded local absolute paths, implement environment variables for security, and perform a directory restructure.

# Step 1: Directory Restructuring
1. Rename the existing `web` directory to `frontend`.
2. Update any internal references, build scripts, or `vite.config.js` settings that might be looking for the old `web` folder.

# Step 2: Environment Variable Implementation (.env)
1. Ensure a `.env` file exists in the `backend` directory.
2. Extract the hardcoded dataset path from `model.py` and place it in the `.env` file like this:
   `DATASET_DIR="C:\Users\USER\.gemini\antigravity\scratch\AICTE-internship-Garbage-Classification\backend\TrashType_Image_Dataset"`
3. Add any other necessary production variables to the `.env` template (e.g., `MODEL_PATH=best_model_finetuned224.keras`, `PORT=10000`).
4. Ensure `.env` is added to the `.gitignore` file so it is never pushed to GitHub.

# Step 3: Code Sanitization (`model.py` and others)
1. Scan `model.py`, `main.py`, and any other backend scripts for the hardcoded path: `r"C:\Users\USER\.gemini\antigravity\scratch\AICTE-internship-Garbage-Classification\backend\TrashType_Image_Dataset"`.
2. Completely remove this hardcoded path.
3. Replace it using the `python-dotenv` package and the `os` module. 
   - Example implementation: `dataset_dir = os.getenv("DATASET_DIR")`
4. Add graceful error handling: If `os.getenv("DATASET_DIR")` is None, log a clear warning that the environment variable is missing.

# Step 4: Render Production Readiness Audit
Review the backend setup for Render deployment compatibility:
1. **Port Binding:** Ensure FastAPI/Uvicorn is configured to bind to `0.0.0.0` and listens to the port dynamically assigned by Render (`os.getenv("PORT", 8000)`).
2. **CORS:** Ensure `CORSMiddleware` in `main.py` is configured properly. While `["*"]` is okay for dev, add a note on how to restrict this to the frontend URL in production.
3. **Dependencies:** Verify `requirements.txt` contains `python-dotenv`, `uvicorn`, and `fastapi`.
4. **Commands:** Provide the exact "Build Command" (e.g., `pip install -r requirements.txt`) and "Start Command" (e.g., `uvicorn main:app --host 0.0.0.0 --port $PORT`) I will need to paste into the Render dashboard.

# Deliverables
Provide the updated code for `model.py` and `main.py` reflecting the `.env` changes, the exact text to put inside my `.env` and `.gitignore` files, and the Render deployment commands.