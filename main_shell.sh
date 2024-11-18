#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/data_preparators/data_preparation_train.py" --raw-data "$1" --generated-dataset "$2"

ontologies=("CC" "MF" "BP")
t_values=(700 900)


for ont in "${ontologies[@]}"; do
    for t in "${t_values[@]}"; do
        #echo $2
        source "$SCRIPT_DIR/gipa_wide_deep_model/run_deep_wide.sh"  "$2/whole_graph_data_${ont}_${t}.pt"
    done
done
