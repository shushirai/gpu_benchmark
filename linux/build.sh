#! /bin/bash

docker build  -f Dockerfile_tf -t tf_bm .
docker build -f Dockerfile_pytorch -t pytorch_bm .