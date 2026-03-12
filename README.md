# Fault Injection Tool for the Reliability Assessment of Deep Learning Algorithms

This is the code repository for the paper **An Effective Iterative Statistical Fault Injection Methodology for Deep Neural Networks**, published on IEEE Transactions on Computers ([project page](https://github.com/cad-polito-it/IterativeFI), [ieee](https://ieeexplore.ieee.org/document/10985784)).


## Overview
**IterativeFI** is an open-source software designed to test the resilience of deep learning algorithms against the occurrence of random-hardware faults. The intent of the framework is to execute advanced statistical fault injection analyses by extending the available and known fault models in the literature.

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

`git clone https://github.com/cad-polito-it/IterativeFI.git`

## Creating a Python Environment
It is recommended to create a virtual environment to manage your dependencies. You can do this using venv:

`python3 -m venv environment_name`

`source environment_name/bin/activate`

## Installing Dependencies
Once your virtual environment is activated, install the required packages listed in requirements.txt:

`pip install -r requirements.txt`

# Usage

The `test` folder contains some sample scripts you can run to test the iterative or one-step approach on CNNs and MLPs. Alternatively, you can run the individual scripts in the main folder using the appropriate parameters.

## One step SFI

To run the **One step SFI** run the following command:

```bash

python one_step_fi.py -dataset mnist -dataset_path ./data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -layer_name conv1 -results_path ./results -e_goal 0.01 -conf 0.99 -p 0.5 -seed 0

```

where:

- `-data`: name of the dataset to use (e.g., `mnist`)
- `-root`: path to the dataset root directory
- `-net`: network architecture to load
- `-weights_path`: path to the pretrained model weights file
- `-layer_name`: name of the target layer where the fault injection analysis will be applied (use `None` to apply it to the full network)
- `-results_path`: directory where results will be saved
- `-e_goal`: target error threshold for the SFI procedure
- `-conf`: confidence level required for the estimation
- `-p`: probability parameter used during the sampling process
- `-seed`: random seed for reproducibility


After execution, the expected output contains a statistical summary of the One Step SFI experiment. The output is organized as following:

```txt
[PROP/wald] K=1  FRcrit_avg=0.00275632  conf=0.99  n=16590
```

- `FRcrit_avg` is the estimated average critical fault rate (proportion of injections that cause a critical failure).
- `conf` is the selected confidence level.
- `n` is the total number of performed injections.


```txt
half_norm=0.00104855  e_goal=0.01000000  FPC=off  DESIGN=WR  N_pop=54420480
```

- `half_norm` is the half-width of the confidence interval (margin of error).
- `e_goal` is the target estimation error specified by the user.
- `FPC` indicates whether Finite Population Correction is applied.
- `DESIGN=WR` denotes sampling with replacement (WR) or without replacement (WOR).
- `N_pop` is the size of the theoretical injection population.

```txt
EP_control=wald  E_wald_final=0.00104855
Wald_CI(99%):   [0.00170777, 0.00380487]   half=0.00104855   se=0.00040705   FPC=off
```

This is the Wald confidence interval for the critical fault rate:

- [`lower`, `upper`] are the confidence bounds.
- `half` is the interval half-width.
- `se` is the standard error of the proportion estimator.

```txt
SampleStdDev_FRcrit_across_injections: s=0.03727662 (n=16590)
injections_used=16590  pilot=16590
```

- `s` is the sample standard deviation across injections.
- `injections_used` is the number of injections effectively used for estimation.
- `pilot` corresponds to the number injections effectively used.


```txt
baseline_pred_dist=[0.098, 0.114, 0.1037, 0.1014, 0.0988, 0.0897, 0.0952, 0.1026, 0.0969, 0.0997]
global_summary_over_injections: fault_pred_dist = [0.09789007836045811, 0.11411031946955998, 0.10371057866184448, 0.10130705244122966, 0.09897595539481616, 0.08965216395418928, 0.09511476190476191, 0.10269871609403255, 0.09700842073538277, 0.09953195298372514] Δmax = 0.000 KL = 0.000 TV = 0.001 H_baseline = 2.301 H_global = 2.301 ΔH = 0.000 BER = 0.0028 per_class = [0.0028305716499981547, 0.0022834512441441157, 0.0029156879601809596, 0.002686737691606241, 0.0026857552840231056, 0.002951973727978131, 0.0026365609534953222, 0.0027104799034623596, 0.0029540841430953906, 0.0029922800347999996] agree = 0.997 flip_asym = 0.230
```

- `baseline_pred_dist` is the prediction distribution without faults.
- `fault_pred_dist` is the global average prediction distribution across all injections.

The following metrics quantify distributional shifts:

- `Δmax`: maximum per-class probability shift.
- `KL`: Kullback–Leibler divergence.
- `TV`: Total Variation distance.
- `H_baseline`, `H_global`: entropy values.
- `ΔH`: entropy variation.
- `BER`: observed bit error rate.
- `per_class`: per-class critical fault rate.
- `agree`: agreement rate between fault-free and faulty predictions.
- `flip_asym`: asymmetry in class flips.


```txt
[E_seq] n, E_wald
16590,0.00104855

[E_final] E_wald=0.00104855
```
These lines report the evolution and final value of the Wald estimation error during the sampling process.

```txt
Top-100 worst injections
  1) Inj   9108 | FRcrit=0.913400 | maj=3@0.39 Δmax=0.290 KL=0.806 | conv3 [25, 27, 2, 2] bit 30
  2) Inj  15104 | FRcrit=0.910500 | maj=3@0.63 Δmax=0.528 KL=1.342 | conv3 [44, 43, 2, 2] bit 30
  3) Inj   1069 | FRcrit=0.903100 | maj=8@1.00 Δmax=0.903 KL=2.334 | conv2 [45, 24, 2, 0] bit 30
  ...
```
This section lists the injections that produced the highest critical fault rate.  
For each injection, the output reports:

- Injection identifier.
- Observed critical fault rate.
- Majority class and its probability.
- Distribution shift metrics (`Δmax`, `KL`).
- Target layer, tensor coordinates, and flipped bit.

These entries highlight the most sensitive locations in the network.

## Iterative SFI

To run the **Iterative SFI** run the following command:

```bash

python iterative_fi.py -dataset mnist -dataset_path ./data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -layer_name conv1 -results_path ./results -e_start 0.05 -e_goal 0.01 -conf 0.99 -p0 0.5 -seed 0

```

where:

- `-data`: name of the dataset to use (e.g., `mnist`)
- `-root`: path to the dataset root directory
- `-net`: network architecture to load
- `-weights_path`: path to the pretrained model weights file
- `-layer_name`: name of the target layer where the fault injection analysis will be applied (use `None` to apply it to the full network)
- `-results_path`: directory where results will be saved
- `-e_start`: initial estimation error used to bootstrap the iterative procedure
- `-e_goal`: target estimation error to be reached before stopping
- `-conf`: required confidence level for the statistical estimation
- `-p0`: initial proportion estimate used for sample size computation
- `-seed`: random seed for reproducibility

The output structure and reported metrics are the same as described in the **One Step SFI** section, including proportion estimation, confidence intervals, distributional analysis, sequential error monitoring, and worst-case injection reporting. The only difference is in the `[E_seq]` section, where multiple lines may appear, each corresponding to a different iteration of the procedure and showing how the estimation error evolves progressively until the target `e_goal` is reached.

```txt
  [PROP/wald] K=1  FRcrit_avg=0.00205301  conf=0.99  steps=1  n=664
  half_norm=0.00452492  e_goal=0.01000000  FPC=off  DESIGN=WR  N_pop=54420480
  EP_control=wald  E_wald_final=0.00452492
  Wald_CI(99%):   [0.00000000, 0.00657794]   half=0.00452492   se=0.00175657   FPC=off
  SampleStdDev_FRcrit_across_injections: s=0.02900720 (n=664)
  injections_used=664  pilot=664  budget_cap=None
  baseline_pred_dist=[0.098, 0.114, 0.1037, 0.1014, 0.0988, 0.0897, 0.0952, 0.1026, 0.0969, 0.0997]
  global_summary_over_injections: fault_pred_dist = [0.0978539156626506, 0.1149875, 0.10420798192771084, 0.10122756024096385, 0.09854894578313253, 0.08949683734939759, 0.09499472891566266, 0.10265376506024096, 0.09661234939759036, 0.09941641566265061] Δmax = 0.001 KL = 0.000 TV = 0.002 H_baseline = 2.301 H_global = 2.301 ΔH = 0.000 BER = 0.0021 per_class = [0.0014906565035652816, 0.0012352039737898964, 0.00156702025072324, 0.001702074570471234, 0.002541034583678845, 0.0022649124927804866, 0.002159372785258682, 0.001947849879048357, 0.002968530468623721, 0.0028458870587666613] agree = 0.998 flip_asym = 0.836

  [E_seq] n, E_wald
  664,0.00452492

  [E_final] E_wald=0.00452492

  Top-100 worst injections (proposed iter EP)
    1) Inj      1 | FRcrit=0.562400 | maj=1@0.68 Δmax=0.562 KL=0.930 | fc1 [114, 461] bit 30
    2) Inj     94 | FRcrit=0.430200 | maj=2@0.53 Δmax=0.430 KL=0.600 | fc1 [186, 5931] bit 30
    3) Inj    227 | FRcrit=0.186800 | maj=1@0.30 Δmax=0.187 KL=0.140 | fc1 [105, 936] bit 30
  ...
```

# Acknowledgments

This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them.

# Main Contributors

- Annachiara Ruospo (annachiara.ruospo@polito.it)
- Lorenzo Fezza (lorenzo.fezza@polito.it)
- Federica Terramagra (federica.terramagra@polito.it)
