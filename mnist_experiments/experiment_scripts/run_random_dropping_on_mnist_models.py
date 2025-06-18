#%%
import os
from functools import partial

import pickle as pkl
import torch.utils
import torch.utils.data
import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from models_and_data.mnist import (
    construct_mnist_components,
    MNIST_NUM_CLASSES
)
from soft_moe.soft_moe_model import SoftMoeModel
from soft_moe.soft_moe_utils import (
    make_phi,
    get_model_predictions_on_loader
)
from soft_moe.soft_moe_utils import get_model_predictions_on_input_with_mask
from models_and_data.mnist_model_config import args
#%%

MODEL_ROOT_DIR = "logs/smoe_mnist_constant_params"
USE_EP = 14
DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:0")
NUM_CLASSES = MNIST_NUM_CLASSES
BATCH_SIZE = 5000

ALL_RESULTS_DICT = {}

#%%
for ne in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    print('='*80)
    print(f"Running for {ne} experts")
    ALL_RESULTS_DICT[ne] = None
    args.num_experts = ne
    args.expert_fn_multiplier = 4/ne
    exp_name = f"ne_{ne}"

    # %% Load model and make components
    log_dir = os.path.join(MODEL_ROOT_DIR, exp_name)
    assert os.path.exists(log_dir), f"Path {log_dir} does not exist"
    comps = construct_mnist_components(args)
    patch_fn = comps["patch_fn"]
    encoder = comps["encoder"]
    decoder = comps["decoder"]
    te_loader = comps["te_loader"]
    phi = make_phi(args, DEVICE)
    expert_fn = comps["expert_fn"]
    expert_list = [expert_fn().to(DEVICE) for _ in range(args.num_experts)]
    smoe = SoftMoeModel(
        num_experts=args.num_experts,
        num_slots=args.num_slots,
        expert_pred_dim=args.expert_out_dim,
        tokenizer=patch_fn,
        encoder=encoder,
        decoder=decoder,
        experts=expert_list,
        phi=phi,
        args=args,
    )
    smoe.load(log_dir, f"ep{USE_EP}")
    smoe.to(DEVICE)
    # %%
    orig_pred_out = get_model_predictions_on_loader(smoe, te_loader, DEVICE)
    all_x = orig_pred_out["x"]
    all_y = orig_pred_out["y"]
    orig_acc = np.mean(orig_pred_out['pred'].argmax(1) == all_y)
    
    # %%
    GPU_DEVICE = torch.device("cuda")
    smoe.to(GPU_DEVICE)
    all_x = torch.from_numpy(all_x).to(GPU_DEVICE)
    all_y = torch.from_numpy(all_y).to(GPU_DEVICE)
    #%%
    # drop_cand_list = list(set([1, ne // 4, ne // 2, (3 * ne // 4), ne - 1]))
    drop_cand_list = list(np.arange(1, ne))
    if 0 in drop_cand_list:
        drop_cand_list.remove(0)
    drop_cand_arr = np.sort(np.array(drop_cand_list))

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(all_x, all_y), batch_size=BATCH_SIZE, shuffle=False
    )
    collect_random_results = {0: orig_acc}
    for num_random_drop in tqdm.tqdm(drop_cand_arr):
        collect_random_results[num_random_drop] = []
        for rand_i in range(10):
            np.random.seed(rand_i)
            torch.manual_seed(rand_i)
            collect_pred = []
            collect_label = []
            for x, y in val_loader:
                x = x.to(GPU_DEVICE)
                y = y.to(GPU_DEVICE)
                mask = np.stack([
                    np.random.choice(ne, size=num_random_drop, replace=False)
                    for _ in range(x.shape[0])
                ])
                h_pred, _, _, _ = get_model_predictions_on_input_with_mask(
                    smoe=smoe, 
                    x=x,
                    mask=torch.from_numpy(mask).to(GPU_DEVICE),
                    return_y_tilde_ranks=False,
                    return_y_tilde=False,
                )
                collect_pred.append(h_pred.copy())
                collect_label.append(y.detach().cpu().numpy().copy())
            collect_pred = np.concatenate(collect_pred, axis=0)
            collect_label = np.concatenate(collect_label, axis=0)
            acc = np.mean(collect_pred.argmax(1) == collect_label)
            
            collect_random_results[num_random_drop].append(acc)
        print(collect_random_results[num_random_drop])
    print(collect_random_results)
    ALL_RESULTS_DICT[ne] = collect_random_results

    pkl.dump(ALL_RESULTS_DICT, open("mnist_random_drop_multiple_trials_results.pkl", "wb"))

# %%
