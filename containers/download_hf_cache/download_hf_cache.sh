#!/bin/bash

apptainer run \
    --nv \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env HF_HOME="${HF_HOME}" \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" containers/download_hf_cache/download_resources.py

echo "Downloads complete! Cache at: ${HF_HOME}"