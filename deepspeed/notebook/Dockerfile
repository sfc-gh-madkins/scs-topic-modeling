FROM mosaicml/composer:0.15.1

# Install JupyterLab
RUN pip3 install jupyterlab

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Lab on port 8888 when the container launches
jupyter labextension enable @jupyter-widgets/jupyterlab-manager

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

deepspeed --num_gpus=8 \
    --module training.trainer \
    --input-model EleutherAI/pythia-6.9b \
    --deepspeed a100_config.json \
    --epochs 1 \
    --local-output-dir /dolly/miles \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --logging-steps 1 \
    --save-steps 1 \
    --save-total-limit 1 \
    --eval-steps 1 \
    --warmup-steps 1 \
    --test-size 1 \
    --lr 5e-6 \
    --bf16 true
