---
title: Garbage Classification Backend
emoji: ♻️
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# Garbage Classification Backend API

This is the FastAPI backend for the Garbage Classification application, deployed on Hugging Face Spaces using Docker.

## 🚀 Features
* FastAPI powered REST API
* Custom CNN & Fine-Tuned MobileNetV2 Models for Garbage/Trash Classification
* Deployed inside a non-root secure container on port 7860

## 🐳 Running Locally
You can run this container locally by executing:
```bash
docker build -t garbage-classifier-backend .
docker run -p 7860:7860 garbage-classifier-backend
```
