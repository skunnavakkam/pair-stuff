# ---
# args: ["--timeout", 10]
# ---

# ## Overview
#
# Quick snippet showing how to connect to a Jupyter notebook server running inside a Modal container,
# especially useful for exploring the contents of Modal Volumes.
# This uses [Modal Tunnels](https://modal.com/docs/guide/tunnels#tunnels-beta)
# to create a tunnel between the running Jupyter instance and the internet.
#
# If you want to your Jupyter notebook to run _locally_ and execute remote Modal Functions in certain cells, see the `basic.ipynb` example :)

import os
import subprocess
import time

import modal

app = modal.App(
    image=modal.Image.debian_slim(python_version="3.9").pip_install(
        "faiss-gpu", "transformer_lens", "matplotlib", "pynndescent", "jupyter"
    )
)
volume = modal.Volume.from_name(
    "modal-examples-jupyter-inside-modal-data", create_if_missing=True
)

CACHE_DIR = "/root/cache"
JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!


@app.function(concurrency_limit=1, volumes={CACHE_DIR: volume}, timeout=7200, gpu="T4")
def run_jupyter(timeout: int):
    jupyter_port = 8888
    import os

    os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
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

        print(f"Jupyter available at => {tunnel.url}/?token={JUPYTER_TOKEN}")

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
def main():
    # Write some images to a volume, for demonstration purposes.
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=7200)


# Doing `modal run jupyter_inside_modal.py` will run a Modal app which starts
# the Juypter server at an address like https://u35iiiyqp5klbs.r3.modal.host.
# Visit this address in your browser, and enter the security token
# you set for `JUPYTER_TOKEN`.
