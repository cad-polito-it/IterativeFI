#!/usr/bin/env bash

# python one_step_fi.py -data cifar10 -root ../data -net cifar10_resnet20 -weights_path ./networks/weights/best_cifar_resnet20.pt -results_path ./results -e_goal 0.01 -conf 0.99
# python one_step_fi.py -data cifar10 -root ../data -net cifar10_mobilenetv2 -weights_path ./networks/weights/best_cifar_mobilenetv2.pt -results_path ./results -e_goal 0.01 -conf 0.99

# python iterative_fi.py -data cifar10 -root ../data -net cifar10_resnet20 -weights_path ./networks/weights/best_cifar_resnet20.pt -results_path ./results -e_start 0.05 -e_goal 0.01 -conf 0.99
# python iterative_fi.py -data cifar10 -root ../data -net cifar10_mobilenetv2 -weights_path ./networks/weights/best_cifar_mobilenetv2.pt -results_path ./results -e_start 0.05 -e_goal 0.01 -conf 0.99

python one_step_fi.py -data cifar10 -root ../data -net cifar10_resnet20 -weights_path ./networks/weights/best_cifar_resnet20.pt -results_path ./results -e_goal 0.01 -conf 0.99
python one_step_fi.py -data cifar10 -root ../data -net cifar10_mobilenetv2 -weights_path ./networks/weights/best_cifar_mobilenetv2.pt -results_path ./results -e_goal 0.01 -conf 0.99

python iterative_fi.py -data cifar10 -root ../data -net cifar10_resnet20 -weights_path ./networks/weights/best_cifar_resnet20.pt -results_path ./results -e_start 0.05 -e_goal 0.01 -conf 0.99
python iterative_fi.py -data cifar10 -root ../data -net cifar10_mobilenetv2 -weights_path ./networks/weights/best_cifar_mobilenetv2.pt -results_path ./results -e_start 0.05 -e_goal 0.01 -conf 0.99