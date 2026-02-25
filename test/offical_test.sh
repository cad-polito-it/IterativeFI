#!/usr/bin/env bash

WEIGHTS_PATH="./networks/weights/best_mnist_cnn.pt"

mapfile -t LAYER_NAMES < <(python3 ./utils/extract_layers.py -weights_path $WEIGHTS_PATH)

for i in 0.01 0.001 0.0001
do
    python one_step_fi.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results -e_goal $i -conf 0.99
    python iterative_fi.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results -e_start 0.05 -e_goal $i -conf 0.99
    
    for layer in "${LAYER_NAMES[@]}"; do
        echo "$layer"
        python one_step_fi.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results -e_goal $i -conf 0.99 -layer_name $layer
        python iterative_fi.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results -e_start 0.05 -e_goal $i -conf 0.99 -layer_name $layer
    done    
done


# WEIGHTS_PATH="./networks/weights/best_banknote_mlp.pt"

# mapfile -t LAYER_NAMES < <(python3 ./utils/extract_layers.py -weights_path $WEIGHTS_PATH)

# for i in 0.01 0.001 0.0001
# do
#     python one_step_fi.py -data banknote -root ../data -net banknote_mlp -weights_path ./networks/weights/best_banknote_mlp.pt -results_path ./results -e_goal $i -conf 0.99
#     python iterative_fi.py -data banknote -root ../data -net banknote_mlp -weights_path ./networks/weights/best_banknote_mlp.pt -results_path ./results -e_start 0.05 -e_goal $i -conf 0.99
    
#     for layer in "${LAYER_NAMES[@]}"; do
#         echo "$layer"
#         python one_step_fi.py -data banknote -root ../data -net banknote_mlp -weights_path ./networks/weights/best_banknote_mlp.pt -results_path ./results -e_goal $i -conf 0.99 -layer_name $layer
#         python iterative_fi.py -data banknote -root ../data -net banknote_mlp -weights_path ./networks/weights/best_banknote_mlp.pt -results_path ./results -e_start 0.05 -e_goal $i -conf 0.99 -layer_name $layer
#     done    
# done