
## table 1：full data

| Method | Model | AUROC | AUPRC | Max MCC | Max F1-score |
|------|------|---------|--------|---------|--------------|
| knowledge-guided and physics-based | AutoDock Vina | 0.5242 | 0.5379 | 0.0731 | 0.6699 |
| statistical machine learning | ChemBoost | 0.8970 | 0.9118 | 0.6476 | 0.8254 |
| supervised representation learning | MolTrans | 0.8774 | 0.8872 | 0.6155 | 0.8070 |
| supervised representation learning | DrugBAN | 0.9058 | 0.9134 | 0.6825 | 0.8377 |
| protein-side foundation model | ConPLex | 0.9152 | 0.9304 | 0.6938 | 0.8414 |
| molecule-side foundation model | MT-DTI | 0.9096 | 0.9250 | 0.7014 | 0.8387 |
| dual-branch foundation model | DTIAM | 0.9470 | 0.9538 | 0.7770 | 0.8865 |

## table 2：unseen drug

| Method | Model | AUROC | AUPRC | Max MCC | Max F1-score |
|------|------|---------|--------|---------|--------------|
| knowledge-guided and physics-based | AutoDock Vina | 0.5361 | 0.5569 | 0.0713 | 0.6858 |
| statistical machine learning | ChemBoost | 0.8306 | 0.8658 | 0.5529 | 0.7664 |
| supervised representation learning | MolTrans | 0.8456 | 0.8663 | 0.5444 | 0.7816 |
| supervised representation learning | DrugBAN | 0.8781 | 0.8967 | 0.6229 | 0.8124 |
| protein-side foundation model | ConPLex | 0.8892 | 0.9090 | 0.6222 | 0.8101 |
| molecule-side foundation model | MT-DTI | 0.8810 | 0.9076 | 0.6208 | 0.8104 |
| dual-branch foundation model | DTIAM | 0.9130 | 0.9318 | 0.6886 | 0.8453 |

## table 3：unseen protein

| Method | Model | AUROC | AUPRC | Max MCC | Max F1-score |
|------|------|---------|--------|---------|--------------|
| knowledge-guided and physics-based | AutoDock Vina | 0.5505 | 0.5368 | 0.1119 | 0.6512 |
| statistical machine learning | ChemBoost | 0.6072 | 0.5944 | 0.1831 | 0.6129 |
| supervised representation learning | MolTrans | 0.6608 | 0.6808 | 0.2940 | 0.6576 |
| supervised representation learning | DrugBAN | 0.6521 | 0.6555 | 0.2492 | 0.6511 |
| protein-side foundation model | ConPLex | 0.8451 | 0.8667 | 0.5849 | 0.7633 |
| molecule-side foundation model | MT-DTI | 0.7628 | 0.7954 | 0.4537 | 0.6797 |
| dual-branch foundation model | DTIAM | 0.8962 | 0.9038 | 0.6574 | 0.8201 |
