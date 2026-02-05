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
skglm
scikit-survival
```

# Installation Guide
conda create -n FreeSurv python=3.8

source activate FreeSurv

pip install -r requirements.txt

- This takes several mins to build

# Run demo

## omics data

**HCC dataset**
python3 Realdata_exps.py --dataset HCC --l1_lambda 0.05 --elastic_lambda=0.1 --FreeSurv_lambda 0.5

**BreastCancer dataset**
python3 Realdata_exps.py --dataset BreastCancer --l1_lambda 0.05 --elastic_lambda 0.1 --FreeSurv_lambda 1

**PDAC dataset**
python3 Realdata_exps.py --dataset PDAC --l1_lambda 0.1 --elastic_lambda 0.2 --FreeSurv_lambda 0.2

## simulated data

""Uncorrelated data""
python3 Simulated_exps.py --exps_class uncorrelated --n_features 256 --n_samples 1000 --ind True --top_n_select 4 --mode Cox-additive --l1_lambda 0.1 --elastic_lambda 0.1 --FreeSurv_lambda 1

""Correlated data""
python3 Simulated_exps.py --exps_class correlated --n_samples 1000 --ind True --top_n_select 4 --mode Cox-additive --l1_lambda 0.01 --elastic_lambda 0.01 --FreeSurv_lambda 1

- The expected running time is from several seconds to mins depends on the number of samples.

# License
This project is licensed under the terms of the MIT license.
