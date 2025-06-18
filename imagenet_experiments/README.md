# Soft MoE adaptation of ViT

This is the codebase we used to run the ImageNet-1k experiments. 
For implementation of the model, we modified the publicly available PyTorch implementation which is available at https://github.com/bwconrad/soft-moe.
To train the model, we have relied on the DeiT repository, which is also publicly available at https://github.com/facebookresearch/deit.

Please refer to their original repository for instructions on installation.


## Code Overview

`smoe-vit` contains the implementation of the model, and `deit` contains the training and evaluation code. 

`smoe-vit` must be installed first for `deit` to run.


Within `deit`, we have included code to 1) train the suite of Soft MoE models, and 2) run the expert subset selection experiments. 


We have included the scripts to train the suite of models for the CIFAR100 experiments. These files are 
- `deit/ls_ne2_mr32.sh`
- `deit/ls_ne8_mr8.sh`
- `deit/ls_ne16_mr4.sh`
- `deit/ls_ne64_mr1.sh`
- `deit/ls_ne128_mr0.5.sh`

The experiment code is in the following file:
- `deit/run_drop_experiment.py`

