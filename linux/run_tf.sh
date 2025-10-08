#! /bin/bash

docker run --gpus all -it --rm \
        -v $(pwd):/workspace tf_bm \
        bash
