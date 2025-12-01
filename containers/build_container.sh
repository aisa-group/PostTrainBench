#!/bin/bash
container="${1}"

export POST_TRAIN_BENCH_CONTAINERS_DIR=${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}
export APPTAINER_BIND=""

apptainer build "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${container}.sif" "containers/${container}.def"
