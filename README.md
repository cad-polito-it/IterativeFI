# Fault Injection Tool for the Reliability Assessment of Deep Learning Algorithms

DA METTERE I LINK
This is the code repository for the paper **An Effective Iterative Statistical Fault Injection Methodology for Deep Neural Networks**, published on IEEE Transactions on Computers ([project page](), [pdf](), [arXiv](), [poster]()).


## Overview
**NOME DELLA REPO CHE DECIDEREMO** is an open-source software designed to test the resilience of deep learning algorithms against the occurrence of random-hardware faults. The intent of the framework is to execute advanced statistical fault injection analyses by extending the available and known fault models in the literature.

## Projects structure

This project is structured as follows:
- `requirements.txt`: Packages to install in a virtual environment to run the application
- `iterative_fi.py`: It performs the Iterative statistical FI campaign, saving relevant statistics over iterations
- `one_step_fi.py`: It performs the One Step statistical FI campaign, saving relevant statistics of the campaign
- `datasets/`: Contains the dataloaders getter function 
- `injector/`: Contains the fault injector for FI campaigns on a single bit of the model parameters
- `networks/`: Contains the networks getter function 
- `results/`: Contains results of the FI campaigns
- `test/`: Contains example scripts for testing the FI campaigns
- `utils/`: Contains utility functions and helper modules for faults generation and sampling


# Setup

To get started, first clone the repository from GitHub:

`git clone https://github.com/DA_CAMBIARE`

## Creating a Python Environment
It is recommended to create a virtual environment to manage your dependencies. You can do this using venv:

`python3 -m venv environment_name`

`source environment_name/bin/activate`

## Installing Dependencies
Once your virtual environment is activated, install the required packages listed in requirements.txt:

`pip install -r requirements.txt`

# Usage

The `test` folder contains some sample scripts you can run to test the iterative or one-step approach on the `resnet20` and `mobilenetv2` networks. Alternatively, you can run the individual scripts in the main folder using the appropriate parameters.

## One step SFI

Spiegare i parametri:

```python

python one_step_fi.py -data cifar10 -root ../data -net cifar10_resnet20 -weights_path ./networks/weights/best_cifar_resnet20.pt -results_path ./results -e_goal 0.01 -conf 0.99 -p 0.5

```
e dire che viene prodotto un output come segue:

```txt
[PROP/wald] K=1  FRcrit_avg=0.06183130  conf=0.99  n=16590
half_norm=0.00481690  e_goal=0.01000000  FPC=off  DESIGN=WR  N_pop=9324032
EP_control=wald  E_wald_final=0.00481690
Wald_CI(99%):   [0.05701440, 0.06664820]   half=0.00481690   se=0.00186991   FPC=off
SampleStdDev_FRcrit_across_injections: s=0.10653219 (n=16590)
injections_used=16590  pilot=16590
baseline_pred_dist=[0.06640625, 0.140625, 0.09765625, 0.1015625, 0.0859375, 0.08203125, 0.125, 0.09375, 0.08984375, 0.1171875]
global_summary_over_injections: fault_pred_dist = [0.06903961347197107, 0.1434909960819771, 0.09555596368294153, 0.09619075874020494, 0.08544963080168777, 0.08635826363773358, 0.12792650881555154, 0.09289246157323688, 0.08785295170283303, 0.11524285149186257] Δmax = 0.005 KL = 0.000 TV = 0.013 H_baseline = 2.281 H_global = 2.281 ΔH = 0.000 BER = 0.0634 per_class = [0.03482608233166684, 0.0224717031679057, 0.0872622061482821, 0.15072332730560578, 0.06074031453778289, 0.07030339561985131, 0.03762620554550934, 0.03576702833031947, 0.08422307833425059, 0.05050632911392405] agree = 0.938 flip_asym = 0.427

[E_seq] n, E_wald
16590,0.00481690

[E_final] E_wald=0.00481690

Top-100 worst injections
  1) Inj   5499 | FRcrit=0.972656 | maj=4@0.67 Δmax=0.582 KL=1.636 | layer3.2.conv1 [62, 8, 1, 2] bit 30
  2) Inj  11779 | FRcrit=0.960938 | maj=4@0.45 Δmax=0.363 KL=1.354 | layer3.1.conv2 [8, 29, 0, 1] bit 30
  3) Inj  15785 | FRcrit=0.953125 | maj=3@0.45 Δmax=0.352 KL=1.143 | layer1.0.conv2 [5, 11, 0, 1] bit 30
  ...
```
Spiegare l'output:

The output of the SFI is stored in the folder `results`. More in details, each result file contains:

The file are named as follow:

## Iterative SFI


Spiegare i parametri:


```python

python iterative_fi.py -data cifar10 -root ../data -net cifar10_resnet20 -weights_path ./networks/weights/best_cifar_resnet20.pt -results_path ./results -e_start 0.05 -e_goal 0.01 -conf 0.99 -p0 0.5

```

e dire che viene prodotto un output come segue:

```txt
[PROP/wald] K=1  FRcrit_avg=0.06134662  conf=0.99  steps=2  n=3844
half_norm=0.00997016  e_goal=0.01000000  FPC=off  DESIGN=WR  N_pop=9324032
EP_control=wald  E_wald_final=0.00997016
Wald_CI(99%):   [0.05137646, 0.07131678]   half=0.00997016   se=0.00387040   FPC=off
SampleStdDev_FRcrit_across_injections: s=0.10415866 (n=3844)
injections_used=3844  pilot=664  block=50  budget_cap=None
baseline_pred_dist=[0.06640625, 0.140625, 0.09765625, 0.1015625, 0.0859375, 0.08203125, 0.125, 0.09375, 0.08984375, 0.1171875]
global_summary_over_injections: fault_pred_dist = [0.06824962604058273, 0.14400689386056192, 0.09529563117845993, 0.09582506828824142, 0.08504528160770031, 0.08605029754162331, 0.12811158623829344, 0.09341160737513007, 0.08891291623309053, 0.11509109163631634] Δmax = 0.006 KL = 0.000 TV = 0.012 H_baseline = 2.281 H_global = 2.280 ΔH = 0.001 BER = 0.0630 per_class = [0.0344310460916937, 0.021404208579026477, 0.08709677419354839, 0.15109461298327062, 0.061169709582820926, 0.07022694613745602, 0.037111407388137355, 0.03483784252514741, 0.08248880242501018, 0.05005202913631634] agree = 0.939 flip_asym = 0.413

[E_seq] n, E_wald
664,0.02329328
3603,0.01032825
3844,0.00997016

[E_final] E_wald=0.00997016

Top-100 worst injections (proposed iter EP)
  1) Inj   1525 | FRcrit=0.968750 | maj=8@0.86 Δmax=0.766 KL=1.883 | layer3.2.conv2 [45, 1, 0, 1] bit 30
  2) Inj   2304 | FRcrit=0.953125 | maj=7@0.62 Δmax=0.523 KL=1.787 | layer3.1.conv2 [48, 41, 2, 0] bit 30
  3) Inj   1816 | FRcrit=0.937500 | maj=3@0.44 Δmax=0.336 KL=1.096 | layer3.1.conv1 [31, 30, 0, 2] bit 30
  ...
```
Spiegare l'output:

The output of the SFI is stored in the folder `results`. More in details, each result file contains:

The file are named as follow:


# Acknowledgments

This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them.