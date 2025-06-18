#!/bin/bash

# Define the range of port numbers to choose from
PORT_MIN=10000
PORT_MAX=65535

# Generate a random port number within the range
PORT=$(shuf -i $PORT_MIN-$PORT_MAX -n 1)

NUM_PROC=$1
shift

torchrun --nproc_per_node=$NUM_PROC --master_port=$PORT train.py "$@"

