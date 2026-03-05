#!/usr/bin/env bash

WEIGHTS_PATH="./networks/weights/GoogLeNet.pt"

# Estrai i layer
mapfile -t LAYER_NAMES < <(python3 ./utils/extract_layers.py -weights_path $WEIGHTS_PATH)

# Definisci i valori di e_goal
E_GOALS=(0.01 0.001 0.0001)

for i in "${!E_GOALS[@]}"; do
    e_goal=${E_GOALS[$i]}
    screen_name="fi_run_$i"

    # Costruisci il comando completo da eseguire nello screen
    CMD="
python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_goal $e_goal -conf 0.99
python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_start 0.05 -e_goal $e_goal -conf 0.99
"

    # Aggiungi i layer
    for layer in "${LAYER_NAMES[@]}"; do
        CMD+="
echo \"$layer\"
python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_goal $e_goal -conf 0.99 -layer_name $layer
python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_start 0.05 -e_goal $e_goal -conf 0.99 -layer_name $layer
"
    done

    # Esegui il comando in uno screen separato
    screen -S "$screen_name" -dm bash -c "$CMD"
done


# WEIGHTS_PATH="./networks/weights/GoogLeNet.pt"

# mapfile -t LAYER_NAMES < <(python3 ./utils/extract_layers.py -weights_path $WEIGHTS_PATH)

# for i in 0.01 0.001 0.0001
# do
#     python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path ./networks/weights/GoogLeNet.pt -results_path ./results -e_goal $i -conf 0.99
#     python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path ./networks/weights/GoogLeNet.pt -results_path ./results -e_start 0.05 -e_goal $i -conf 0.99
    
#     for layer in "${LAYER_NAMES[@]}"; do
#         echo "$layer"
#         python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path ./networks/weights/GoogLeNet.pt -results_path ./results -e_goal $i -conf 0.99 -layer_name $layer
#         python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path ./networks/weights/GoogLeNet.pt -results_path ./results -e_start 0.05 -e_goal $i -conf 0.99 -layer_name $layer
#     done    
# done


# WEIGHTS_PATH="./networks/weights/best_mnist_cnn.pt"

# mapfile -t LAYER_NAMES < <(python3 ./utils/extract_layers.py -weights_path $WEIGHTS_PATH)

# for i in 0.01 0.001 0.0001
# do
#     python one_step_fi.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results -e_goal $i -conf 0.99
#     python iterative_fi.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results -e_start 0.05 -e_goal $i -conf 0.99
    
#     for layer in "${LAYER_NAMES[@]}"; do
#         echo "$layer"
#         python one_step_fi.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results -e_goal $i -conf 0.99 -layer_name $layer
#         python iterative_fi.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results -e_start 0.05 -e_goal $i -conf 0.99 -layer_name $layer
#     done    
# done


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