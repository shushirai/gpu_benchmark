#! /bin/bash

docker run --gpus all -it --rm \
        -v $(pwd):/workspace pytorch_bm \
        bash
