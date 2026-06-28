# Context-Aware Adaptive Federated Learning (CAFL)

## Overview

This repository contains the implementation of the **Context-Aware Adaptive Federated Learning (CAFL)** framework proposed for secure and adaptive federated learning in dynamic IoT environments characterized by non-IID data distributions and context drift.

CAFL enhances traditional federated learning through:

* Context-aware client profiling
* Adaptive client clustering
* Context drift detection and adaptation
* Lightweight fairness-aware aggregation
* Improved robustness in dynamic IoT environments

## Repository Structure

```text
ablation.py              # Ablation studies
baselines.py             # Baseline federated learning methods
data_loader.py           # Dataset loading and preprocessing
experiment.py            # Main experimental pipeline
federated.py             # Federated learning framework
model.py                 # Model definitions
plot_results.py          # Result visualization utilities

Data/
└── README.md            # Dataset placement instructions

results/
├── ablation_results.json
├── experiment_results.json
└── figures/
```

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets used in this study are not distributed within this repository.

Please obtain the datasets from their original sources and place them in the `Data/` directory before running experiments.

Datasets used:

* FLY-SMOTE
* FD-IDS

## Running Experiments

Execute the main experimental pipeline:

```bash
python experiment.py
```

Generate figures and visualizations:

```bash
python plot_results.py
```

Run baseline comparisons:

```bash
python baselines.py
```

Run ablation studies:

```bash
python ablation.py
```

## Results

The repository includes experimental outputs and generated figures used in the evaluation of the proposed CAFL framework.

## Reproducibility

This release corresponds to the version of the code associated with the research paper:

**"Context-Aware Adaptive Federated Learning for Non-IID and Context Drift IoT Environments"**

All experiments can be reproduced using the provided implementation and the referenced datasets.

## License

This project is released for academic and research purposes.
