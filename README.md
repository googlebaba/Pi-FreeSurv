# System Requirements
## Hardware requirements
'Pi-FreeSurv' package requires a standard computer with at least one Nvidia 3090 GPU.

## Software requirements
### OS requirements
This package is supported for *Linux*. The package has been tested on the following system:
+ Linux: Ubuntu 18.04

### Python Dependencies
'Pi-FreeSurv' mainly depends on the Python scientific stack.

```
lifelines
numpy
pandas
scikit-learn
torch
```

# Installation Guide
conda create -n FreeSurv python=3.8

source activate FreeSurv

pip install -r requirements.txt

- This takes several mins to build

# Run demo

## omics data

### FreeSurv
python3 FreeSurv_HCC.py 0.5 #OS
python3 FreeSurv_breast_cancer 1 1 # OS
python3 FreeSurv_breast_cancer 1 0 # RFS

python3 FreeSurv_PDAC 0.2 1 # OS
python3 FreeSurv_PDAC 0.2 0 # RFS

python3 FreeSurv_PBT 0.05  # OS

### Cox lasso and Cox elasticnet
python3 Cox_lasso_HCC.py 0.05 0.1 #OS

python3 Cox_lasso_breast_cancer.py 0.05 0.1 1 #OS
python3 Cox_lasso_breast_cancer.py 0.05 0.1 0 #RFS

python3 Cox_lasso_PDAC 0.1 0.2 1 # OS
python3 Cox_lasso_PDAC 0.1 0.2 0 # RFS

python3 Cox_lasso_PBT 0.05 0.1  # OS




## simulated data

### FreeSurv
python3 FreeSurv_syn.py Cox-additive 1 1000 1

python3 FreeSurv_syn_OOD.py Cox-additive 1000 1

### Cox lasso and Cox elasticnet
python3 Cox_syn_lasso.py Cox-additive 1 1000 0.1 0.1

python3 Cox_syn_lasso_OOD.py Cox-additive 1000 0.01 0.01


- The expected running time is from several seconds to mins depends on the number of samples.

# License
This project is licensed under the terms of the MIT license.
