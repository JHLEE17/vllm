#!/bin/bash

while true; do
    echo "Running bench.sh..."
    bash bench_openai.sh
    
    sleep 10

    echo "bench.sh has finished. Restarting..."
done
