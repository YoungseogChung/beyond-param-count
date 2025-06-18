# Soft MoE adaptation of Astroformer

This is the codebase we used to run the CIFAR10 and CIFAR100 experiments. 
We have adapted the Astroformer model: [Astroformer: More Data Might not be all you need for Classification](https://arxiv.org/abs/2304.05350)
and heavily relied on their codebase for their model implementation, which is publicly available at https://github.com/Rishit-dagli/Astroformer.

Because the training procedure for CIFAR10 and CIFAR100 are identical, we just provide the code to run the CIFAR100 experiments, but running the CIFAR10 experiments only requires changing the dataset.

Please refer to their original repository for instructions on installation.


## Code Overview

We have included code to 1) train the suite of Soft MoE models, and 2) run the expert subset selection experiments. 


We have included the scripts to train the suite of models for the CIFAR100 experiments. These files are 
- `pytorch-image-models/ls_a1_smoe_2_32.sh`
- `pytorch-image-models/ls_a1_smoe_8_8.sh`
- `pytorch-image-models/ls_a1_smoe_16_4.sh`
- `pytorch-image-models/ls_a1_smoe_64_1.sh`
- `pytorch-image-models/ls_a1_smoe_128_0.5.sh`

The experiment code is in the following file:
- `pytorch-image-models/run_drop_experiment.py`

