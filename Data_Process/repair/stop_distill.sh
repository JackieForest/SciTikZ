#!/bin/bash

mkdir -p stop_files

for INDEX in {0..9}
do
    touch stop_files/distill_${INDEX}.flag
    echo "Created stop_files/distill_${INDEX}.flag"
done

echo "All stop flags created."
