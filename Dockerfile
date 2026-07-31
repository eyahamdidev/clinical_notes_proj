# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend-react/package.json frontend-react/package-lock.json ./
RUN npm ci
COPY frontend-react/ .
RUN npm run build

# ---- Stage 2: Python backend, serving the built frontend too ----
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Drop the built React app in as static files FastAPI will serve at "/"
COPY --from=frontend-build /frontend/dist ./static

# HF Spaces (Docker SDK) expects the app on port 7860
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
