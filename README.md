# DermaVision AI

A web app that sorts a dermoscopic image into one of the seven HAM10000 lesion classes and shows
the confidence for each. It is a learning and screening-support tool, not a medical device.

- `backend/`  FastAPI service with a CPU PyTorch model (`POST /predict`)
- `frontend/` Next.js site that calls the API
