# 2026-DTI-DTA-Review

This repository provides a benchmark of **12 DTA/DTI prediction models** evaluated on the **BioSNAP** and **KIBA** datasets.

## Repository Structure

The repository currently contains three folder:

* **`dataset`** — Datasets used in the experiments and the results of all models
* **`model`** — 12 DTA/DTI prediction models
* **`result`** — 12 DTA/DTI prediction values

## Results

The computational experiment results of all prediction models on the BioSNAP and KIBA datasets are shown below:


### Table 1：Performance comparison between seven DTI prediction models on BIOSANP dataset under random-split settings

| Model | AUROC | AUPRC | Max MCC | Max F1-score |
|------|------|------|------|------|
| AutoDock Vina | 0.524 | 0.538 | 0.073 | 0.670 |
| ChemBoost | 0.897 | 0.912 | 0.648 | 0.825 |
| MolTrans | 0.877 | 0.887 | 0.616 | 0.807 |
| DrugBAN | 0.906 | 0.913 | 0.683 | 0.838 |
| ConPLex | 0.915 | 0.930 | 0.694 | 0.841 |
| MT-DTI | 0.910 | 0.925 | 0.701 | 0.839 |
| DTIAM | 0.947 | 0.954 | 0.777 | 0.887 |

### Table 2：Performance comparison between seven DTI prediction models on BIOSANP dataset under unseen-drug settings

| Model | AUROC | AUPRC | Max MCC | Max F1-score |
|------|------|------|------|------|
| AutoDock Vina | 0.536 | 0.557 | 0.071 | 0.686 |
| ChemBoost | 0.831 | 0.866 | 0.553 | 0.766 |
| MolTrans | 0.846 | 0.866 | 0.544 | 0.782 |
| DrugBAN | 0.878 | 0.897 | 0.623 | 0.812 |
| ConPLex | 0.889 | 0.909 | 0.622 | 0.810 |
| MT-DTI | 0.881 | 0.908 | 0.621 | 0.810 |
| DTIAM | 0.913 | 0.932 | 0.689 | 0.845 |

### Table 3：Performance comparison between seven DTI prediction models on BIOSANP dataset under unseen-target settings

| Model | AUROC | AUPRC | Max MCC | Max F1-score |
|------|------|------|------|------|
| AutoDock Vina | 0.551 | 0.537 | 0.112 | 0.651 |
| ChemBoost | 0.607 | 0.594 | 0.183 | 0.613 |
| MolTrans | 0.661 | 0.681 | 0.294 | 0.658 |
| DrugBAN | 0.652 | 0.656 | 0.249 | 0.651 |
| ConPLex | 0.845 | 0.867 | 0.585 | 0.763 |
| MT-DTI | 0.763 | 0.795 | 0.454 | 0.680 |
| DTIAM | 0.896 | 0.904 | 0.657 | 0.820 |

### Table 4：Performance comparison between seven DTA prediction models on KIBA dataset

| Model | MSE | CI | $R_m^2$ |
|---|---|---|---|
| ChemBoost | 0.176 | 0.856 | 0.724 |
| DeepDTAGen | 0.175 | 0.874 | 0.736 |
| DMFF-DTA | 0.157 | 0.881 | 0.757 |
| FusionDTA | 0.149 | 0.884 | 0.757 |
| SubMDTA | 0.140 | 0.892 | 0.767 |
| PMMR | 0.158 | 0.878 | 0.745 |
| DTIAM | 0.157 | 0.876 | 0.731 |
