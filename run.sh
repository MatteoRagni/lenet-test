#!/bin/bash

nvidia-docker run -v $(pwd):/lenet -v /tmp:/tmp -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu bash -c "cd /lenet && python main.py"
