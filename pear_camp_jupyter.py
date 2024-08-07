
import os
import subprocess
import time
import modal

JUPYTER_TOKEN = os.getenv('JUPYTER_TOKEN')
hf_token = os.getenv('HF_TOKEN')

img = modal.Image.debian_slim().pip_install("torch", "jupyter", "transformer_lens", "matplotlib").env(
    {"HF_TOKEN": hf_token}
)

app = modal.App()
CACHE_DIR = "/root/cache"

volume = modal.Volume.from_name(
    "data", create_if_missing=True
)

@app.function(image=img, concurrency_limit=1, volumes={CACHE_DIR: volume}, timeout=10_000, gpu="A100-80gb")
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 10_000):
    # Write some images to a volume, for demonstration purposes.
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)