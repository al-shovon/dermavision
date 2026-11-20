# Free hosting: Vercel (site) + Render (API)

| Part | Where | Cost | Limits to know |
|---|---|---|---|
| Next.js site (`frontend/`) | Vercel, Hobby plan | free, personal and non-commercial use | none that matter here |
| API (`backend/`) | Render, free web service | free, no credit card | 512 MB RAM, 0.1 CPU, sleeps after 15 minutes without traffic, about 1 minute to wake |

The API runs on ONNX Runtime (no PyTorch), which is what makes it fit in 512 MB.

## 0. Export the model once (Colab)

Run the last cell of `01_check_pretrained_model.ipynb`. It exports the model, checks it against PyTorch and downloads
`dermavision.onnx`. Put the file in `backend/weights/dermavision.onnx`.

## 1. Test the ONNX model locally

In `backend/.env` set:

    MODEL_PATH=weights/dermavision.onnx

Restart the API (Ctrl+Shift+B in VS Code) and upload an image on http://localhost:3000. The model name shown under the
result should now be `dermavision.onnx`.

## 2. Put the project on GitHub

Create an empty repository on github.com, then in the `dermavision` folder:

    git init
    git add .
    git commit -m "DermaVision AI"
    git branch -M main
    git remote add origin https://github.com/YOUR-USER/dermavision.git
    git push -u origin main

`.env` files and `node_modules` are ignored. `backend/weights/dermavision.onnx` (about 16 MB) must be committed.

## 3. API on Render

1. render.com, sign in with GitHub, then New > Blueprint, choose the repository. Render reads `render.yaml`.
   (Manual alternative: New > Web Service, Runtime Docker, Dockerfile path `./backend/Dockerfile`,
   Docker build context `./backend`, Instance type Free, Health check path `/health`.)
2. When it asks for `ALLOWED_ORIGINS`, type `http://localhost:3000` for now. You will change it in step 5.
3. Wait for the deploy to finish. Open `https://YOUR-SERVICE.onrender.com/health`. It should show `{"status":"ok", ...}`.
   The first request after a sleep takes about a minute.

## 4. Site on Vercel

1. vercel.com, sign in with GitHub, then Add New > Project, import the same repository.
2. Set **Root Directory** to `frontend`. The framework is detected as Next.js.
3. Under Environment Variables add `NEXT_PUBLIC_API_URL` = `https://YOUR-SERVICE.onrender.com` (no slash at the end).
4. Deploy. Vercel gives you an address such as `https://dermavision-xxxx.vercel.app`.

`NEXT_PUBLIC_` variables are built into the site, so after changing one you must redeploy.

## 5. Allow the site to call the API

In Render, open the service > Environment, set `ALLOWED_ORIGINS` to your Vercel address
(`https://dermavision-xxxx.vercel.app`, several addresses can be separated by commas) and save. Render redeploys.

## 6. Check it

Open the Vercel address and analyze an image. If it says it cannot reach the server, wait a minute (the API is waking up)
and try again. If the browser console shows a CORS error, step 5 is wrong or has not finished.

Before a demo, open the site 2 minutes early so the API is awake.

## What does not work for free

- **Vercel for the API (with PyTorch):** Python functions are limited to 500 MB uncompressed, and Vercel's own docs
  name PyTorch as a package that needs its larger beta limit. Use Vercel for the site only.
- **Hugging Face Docker Spaces:** creating one now needs a paid plan. Static Spaces are free but cannot run the API.
- **Streamlit Community Cloud:** it hosts Streamlit apps only. This project is Next.js plus FastAPI, so it cannot run there.
