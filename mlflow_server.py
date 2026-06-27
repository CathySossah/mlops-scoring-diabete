import os
import subprocess

port = os.environ.get("PORT", "5000")
subprocess.run([
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", port,
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "./mlruns"
])
