# PROTA
Matlab source codes for the Probabilistic Rank-One Tensor Analysis (PROTA) algorithm presented in the paper [Probabilistic Rank-One Tensor Analysis with Concurrent Regularization](https://ieeexplore.ieee.org/document/8718518).

## Usage
Face recognition with PROTA on 2D images from the FERET dataset: 
```
Demo_PROTA.m
```

## Descriptions of the files in this repository  
 - DBpart.mat stores the indices for training (2 samples per class) /test data partition.
 - FERETC80A45.mat stores 320 faces (32x32) of 80 subjects (4 samples per class) from the FERET database.
 - Demo_PROTA.m provides example usage of PROTA for subspace learning and classification on 2D face images.
 - PROTA_MCR.m implements PROTA with moment-based concurrent regularization described as Alg.2 in [paper](https://ieeexplore.ieee.org/document/8718518).
 - projPROTA_MCR.m projects tensors into the subspace learned by PROTA_MCR.
 - PROTA_BCR.m implements PROTA with Bayesian concurrent regularization described as Alg.3 in [paper](https://ieeexplore.ieee.org/document/8718518).
 - projPROTA_BCR.m projects tensors into the subspace learned by PROTA_BCR.
 - sortProj.m sorts features by their Fisher scores in descending order.

## Requirement
[Tensor toolbox v2.6](http://www.tensortoolbox.org/).

## Citation
If you find our codes helpful, please consider cite the following [paper](https://ieeexplore.ieee.org/document/8718518):
```
@article{
    zhou2019PROTA,
    title={Probabilistic Rank-One Tensor Analysis With Concurrent Regularizations},
    author={Yang Zhou and Haiping Lu and Yiu-ming Cheung},
    journal={IEEE Transactions on Cybernetics},
    year={2019},
    doi={10.1109/TCYB.2019.2914316},
}
```
