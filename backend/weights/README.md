# Model weights

Put `dermavision.onnx` here (exported by the last cell of the Colab notebook).
The Docker image and the deployment copy this file, so commit it to Git (it is about 16 MB).

Local development with PyTorch weights: put a `.pth` file here and set `MODEL_PATH` in `.env`.
`*.pth` files are ignored by Git.
