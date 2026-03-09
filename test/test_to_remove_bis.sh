#!/usr/bin/env bash

WEIGHTS_PATH="./networks/weights/GoogLeNet.pt"

# Estrai i layer
# mapfile -t LAYER_NAMES < <(python3 ./utils/extract_layers.py -weights_path $WEIGHTS_PATH)
mapfile -t LAYER_NAMES < <(
    python3 ./utils/extract_layers.py -weights_path "$WEIGHTS_PATH" | grep -v 'bn$'
)

# for layer in "${LAYER_NAMES[@]}"; do
#     echo "$layer"
# done

half=$(( ${#LAYER_NAMES[@]} / 2 ))
second_half=( "${LAYER_NAMES[@]:half}" )

printf '%s\n' "${second_half[@]}"

# E_GOALS=(0.01 0.001 0.0001)
E_GOALS=(0.001)

for i in "${!E_GOALS[@]}"; do
    e_goal=${E_GOALS[$i]}

    screen_one="fi_run_one_${i}_bis"
    screen_iter="fi_run_iter_${i}_bis"

    CMD_ONE=""
    CMD_ITER=""

    resumed=false
    
    # ------------------
    # LAYER RUNS
    # ------------------

    for layer in "${second_half[@]}"; do

        layer_suffix="_layer_${layer}"

        one_file="./results/one_step_net_googlenet_data_cifar10_egoal_${e_goal}${layer_suffix}.txt"
        iter_file="./results/iterative_fi_net_googlenet_data_cifar10_estart_0.05_egoal_${e_goal}${layer_suffix}.txt"

        if [[ -f "$one_file" && -f "$iter_file" ]]; then
            echo "SKIP layer $layer"
            continue
        fi

        [[ $resumed == false ]] && echo "RESUME from layer $layer (egoal=$e_goal)" && resumed=true

        if [[ ! -f "$one_file" ]]; then
            CMD_ONE+="
echo 'RUN one_step layer $layer'
python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_goal $e_goal -conf 0.99 -layer_name $layer
"
        fi

        if [[ ! -f "$iter_file" ]]; then
            CMD_ITER+="
echo 'RUN iterative layer $layer'
python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_start 0.05 -e_goal $e_goal -conf 0.99 -layer_name $layer
"
        fi

    done

    # ------------------
    # START SCREENS
    # ------------------

    if [[ -n "$CMD_ONE" ]]; then
        screen -S "$screen_one" -dm bash -c "$CMD_ONE"
        echo "Started screen $screen_one"
    fi

    if [[ -n "$CMD_ITER" ]]; then
        screen -S "$screen_iter" -dm bash -c "$CMD_ITER"
        echo "Started screen $screen_iter"
    fi

done

# for i in "${!E_GOALS[@]}"; do
#     e_goal=${E_GOALS[$i]}
#     screen_name="fi_run_$i"

#     # Costruisci il comando completo da eseguire nello screen
#     CMD="
# python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_goal $e_goal -conf 0.99
# python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_start 0.05 -e_goal $e_goal -conf 0.99
# "

#     for layer in "${LAYER_NAMES[@]}"; do
#         CMD+="
# echo \"$layer\"
# python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_goal $e_goal -conf 0.99 -layer_name $layer
# python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_start 0.05 -e_goal $e_goal -conf 0.99 -layer_name $layer
# "
#     done

#     screen -S "$screen_name" -dm bash -c "$CMD"
# done


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