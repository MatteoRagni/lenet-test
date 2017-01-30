#!/bin/bash

nvidia-docker run -v /tmp/nn:/nn -it -p 6006:6006 gcr.io/tensorflow/tensorflow:latest-gpu bash -c "tensorboard --purge_orphaned_data --reload_interval 10 --logdir=/nn"
