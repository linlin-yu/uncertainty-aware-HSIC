# Uncertainty-aware Graph-based Hyperspectral Image Classification

This repository presents the experiments of the paper: 

Uncertainty-aware Graph-based Hyperspectral Image Classification<br>
Linlin Yu, Yifei Lou, Feng Chen <br>
International Conference on Learning Representations (ICLR), 2024. 

[[paper](https://openreview.net/forum?id=8dN7gApKm3)]

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
conda activate uhsic
```

## Data Preprocessing & Running Experiments
The experiments include three datasets: PaviaU, KSC and Houston2013.
1. Download the raw files including the feature and classification ground-truth matrix and save them under the folder `raw_data/{dataset}/`；
2. Run an unsupervised unmixing model and get the endmember matrix as the prior knowledge for the proposed architecture; In our paper, we use the '[Blind Hyperspectral Unmixing Based on Graph Total Variation Regularization (https://ieeexplore.ieee.org/document/9200736)]' to generate the predicted abundance matrix and endmember matrix, and note that we need to run a permutation algorithm to match the endmember matrix with the material label (we provide sample code in the `data/find_perm`). The generated matrix should be saved under the folder `data/{dataset}/unmxing/`; Then we need to 
3. Run `data/data_preprocess.py`, which will generate a folder under `data/{dataset}/raw` for required matrices;
4. For 'GKDE' based models, first run `alpha_prior_generation.py` and `probability_teacher_generation.py`, which will generate and save GKDE teacher and probability teacher tensors under folder `teacher`;
5. For experiments related to misclassification detection, please execute the Python files that end with `clearngraph`. For out-of-distribution (OOD) detection experiments, run the Python files ending with `oodgraph`. For experiments involving softmax graph convolutional networks (GCN), execute the Python files that begin with `classification`. For experiments on enhanced GCN (EGCN) models based on Gaussian Kernel Density Estimation (GKDE), run Python files starting with `GKDE`. Lastly, for experiments related to 'GPN' based models, please run Python files beginning with `GPN`.





## Cite
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{
yu2024uncertaintyaware,
title={Uncertainty-aware Graph-based Hyperspectral Image Classification},
author={Linlin Yu and Yifei Lou and Feng Chen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=8dN7gApKm3}
}
```
