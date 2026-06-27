import os
import subprocess
import sys

port = os.environ.get("PORT", "5000")
cmd = [
    sys.executable, "-m", "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", port,
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "./mlruns"
]
subprocess.run(cmd)
