#!/usr/bin/env bash

WEIGHTS_PATH="./networks/weights/GoogLeNet.pt"

mapfile -t LAYER_NAMES < <(
    python3 ./utils/extract_layers.py -weights_path "$WEIGHTS_PATH" | grep -v 'bn$'
)

# for layer in "${LAYER_NAMES[@]}"; do
#     echo "$layer"
# done

E_GOALS=(0.01 0.001 0.0001)

for i in "${!E_GOALS[@]}"; do
    e_goal=${E_GOALS[$i]}
    screen_name="fi_run_$i"

    CMD=""
    resumed=false

    one_file="./results/one_step_net_googlenet_data_cifar10_egoal_${e_goal}.txt"
    iter_file="./results/iterative_fi_net_googlenet_data_cifar10_estart_0.05_egoal_${e_goal}.txt"

    if [[ -f "$one_file" ]]; then
        echo "SKIP one_step base (egoal=$e_goal)"
    else
        [[ $resumed == false ]] && echo "RESUME from base one_step (egoal=$e_goal)" && resumed=true
        CMD+="
echo 'RUN one_step base egoal=$e_goal'
python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_goal $e_goal -conf 0.99
"
    fi

    if [[ -f "$iter_file" ]]; then
        echo "SKIP iterative base (egoal=$e_goal)"
    else
        [[ $resumed == false ]] && echo "RESUME from base iterative (egoal=$e_goal)" && resumed=true
        CMD+="
echo 'RUN iterative base egoal=$e_goal'
python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_start 0.05 -e_goal $e_goal -conf 0.99
"
    fi

    for layer in "${LAYER_NAMES[@]}"; do

        layer_suffix="_layer_${layer}"

        one_file="./results/one_step_net_googlenet_data_cifar10_egoal_${e_goal}${layer_suffix}.txt"
        iter_file="./results/iterative_fi_net_googlenet_data_cifar10_estart_0.05_egoal_${e_goal}${layer_suffix}.txt"

        if [[ -f "$one_file" && -f "$iter_file" ]]; then
            echo "SKIP layer $layer"
            continue
        fi

        [[ $resumed == false ]] && echo "RESUME from layer $layer (egoal=$e_goal)" && resumed=true

        CMD+="
echo 'Processing layer $layer'
"

        if [[ ! -f "$one_file" ]]; then
            CMD+="
echo 'RUN one_step layer $layer'
python one_step_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_goal $e_goal -conf 0.99 -layer_name $layer
"
        fi

        if [[ ! -f "$iter_file" ]]; then
            CMD+="
echo 'RUN iterative layer $layer'
python iterative_fi.py -data cifar10 -root ../data -net googlenet -weights_path $WEIGHTS_PATH -results_path ./results -e_start 0.05 -e_goal $e_goal -conf 0.99 -layer_name $layer
"
        fi

    done

    screen -S "$screen_name" -dm bash -c "$CMD"
done
