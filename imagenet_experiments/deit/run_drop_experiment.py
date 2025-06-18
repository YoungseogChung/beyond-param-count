
import os
import argparse
import copy

import numpy as np
import pickle as pkl
import torch
import torch.backends.cudnn as cudnn
import json

from timm.data import Mixup
from timm.models import create_model
from timm.utils import NativeScaler, get_state_dict, ModelEma

# from datasets import build_dataset
from my_datasets import build_dataset, batch_sampler
from engine import evaluate
import utils

### BEGIN: imports
from collections import OrderedDict
from soft_moe.hf_vision_transformer import (
    hf_soft_moe_vit_tiny,
    hf_soft_moe_vit_small,
    hf_soft_moe_vit_base,
)
from soft_moe.soft_moe_utils import convert_smoe_vit_vmap_state_dict_to_original
### END: imports


def main(args):
    utils.init_distributed_mode(args)
    print(f"Distributed: {args.distributed}")

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    use_collate_fn = None if args.val_subset in [1000, 3000, 5000, 10000] else batch_sampler
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=use_collate_fn,
    )

    ### BEGIN: model additions block
    if args.model_type is not None:
        if args.model_type == "smoe_tiny":
            model_fn = hf_soft_moe_vit_tiny
        elif args.model_type == "smoe_small":
            model_fn = hf_soft_moe_vit_small
        elif args.model_type == "smoe_base":
            model_fn = hf_soft_moe_vit_base
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        model = model_fn(
            num_experts=args.num_experts, 
            moe_mlp_ratio=args.moe_mlp_ratio, 
            num_classes=args.nb_classes
        )
        if hasattr(args, "vmap_model") and args.vmap_model:
            model.make_vmap_model()
        model.forward = model.forward_logits
        print(f"Model total params: {sum(p.numel() for p in model.parameters())}")
        print(f"  - total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"  - expert parameters: {model.num_expert_parameters()}")
    else:
        print(f"Creating model: {args.model}")
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )
    ### END: model additions block
    
    print(f"  - Loading model onto device {device}")
    model.to(device)
    print(f"  - Model loaded on device {device}")

    ### BEGIN: do we need this?
    # model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device='cpu' if args.model_ema_force_cpu else '',
    #         resume='')
    ### END: do we need this?

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  - Number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr


    checkpoint = torch.load(args.resume, map_location='cpu')


    print(f"  - Loading in weights from {args.resume}")
    ### BEGIN: vmap_model -> create base_expert attributes before loading
    # breakpoint()
    if hasattr(args, "vmap_model") and args.vmap_model:
        model_without_ddp.prepare_vmap_model_loading()
        model_without_ddp.load_state_dict(checkpoint['model'])
    ### END: vmap_model -> create base_expert attributes before loading
    else:
        print("    - Converting vmap state_dict to original state_dict")
        orig_model_converted_state_dict = convert_smoe_vit_vmap_state_dict_to_original(
            checkpoint['model'])
        model_without_ddp.load_state_dict(orig_model_converted_state_dict)
    print(f"  - Model weights loaded from {args.resume}")

    all_results = {}
    all_results["forward"] = {}
    all_results["random"] = {}
    all_results["heuristic"] = {}

    """ Begin All Experiments"""

    ###########################################################################
    ### Original 
    ###########################################################################
    if args.run_forward:
        # Original forward
        print("Running original forward")
        forward_test_stats = evaluate(data_loader_val, model, device)
        print(f"  - Original acc on {len(dataset_val)} test images: {forward_test_stats['acc1']:.2f}%")
        all_results["forward"]["drop_none"] = copy.deepcopy(forward_test_stats)

    ### BEGIN: configure drop candidates
    drop_cand_list = sorted(
        set([model.num_experts // 2, 
             3 * model.num_experts // 4, 
             7 * model.num_experts // 8, 
            ]
        )
    )
    if 0 in drop_cand_list:
            drop_cand_list.remove(0)
    print(f"Drop candidates: {drop_cand_list}")
    ### END: configure drop candidates

    model._drop_forward_switch(use_renorm=False)
    for num_drop in drop_cand_list:
        ###########################################################################
        ### Random 10 seeds
        ###########################################################################
        if args.run_random_drop:
            rand_drop_trials_list = []
            for rand_i in range(10):
                torch.manual_seed(rand_i)
                np.random.seed(rand_i)
                # Random drop
                ### BEGIN: get random drop expert indices
                rand_drop_expert_idxs = np.stack(
                    [
                        np.random.choice(model.num_experts, num_drop, replace=False)
                        for _ in range(args.batch_size)
                    ],
                    axis=0
                )
                rand_drop_expert_idxs = torch.from_numpy(rand_drop_expert_idxs).to(device)
                ### END: get random drop expert indices

                ### BEGIN: apply random drop preparations
                heuristic_set_mlp_list, mask_set_mlp_list = model._apply_pre_heuristic(
                    drop_expert_idxs=rand_drop_expert_idxs, 
                    num_heuristic_drop=-9999)
                model._check_pre_heurstic_done(
                    drop_expert_idxs=rand_drop_expert_idxs, 
                    num_heuristic_drop=-9999)
                ### END: apply random drop preparations
                print(f"Random drop with {num_drop} experts dropped, trial {rand_i}")
                random_drop_test_stats = evaluate(data_loader_val, model, device)
                print(f"  - Random drop={num_drop} acc on {len(dataset_val)} test images: {random_drop_test_stats['acc1']:.2f}%")
                rand_drop_trials_list.append(copy.deepcopy(random_drop_test_stats))
            all_results["random"][f"drop_{num_drop}"] = rand_drop_trials_list
        ###########################################################################
        ### Heuristic
        ###########################################################################
        if args.run_heuristic_drop:
            # Heuristic drop
            ### BEGIN: apply heuristic drop preparations
            heuristic_set_mlp_list, mask_set_mlp_list = model._apply_pre_heuristic(
                drop_expert_idxs=None, 
                num_heuristic_drop=num_drop)
            model._check_pre_heurstic_done(
                drop_expert_idxs=None, 
                num_heuristic_drop=num_drop)
            ### END: apply heuristic drop preparations
            print(f"Heuristic drop with {num_drop} experts dropped")
            heuristic_drop_test_stats = evaluate(data_loader_val, model, device)
            print(f"  -  Heuristic drop={num_drop} acc on {len(dataset_val)} test images: {heuristic_drop_test_stats['acc1']:.2f}%")
            all_results["heuristic"][f"drop_{num_drop}"] = copy.deepcopy(heuristic_drop_test_stats)
        
    return all_results

def parse_exp_args():
    parser = argparse.ArgumentParser(description='Run drop experiment')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the model directory')
    # /home/scratch/youngsec/rs/moe/git_packages/deit/checkpoint/youngsec/experiments/smoe_base_ne2_mr32_vmap
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the checkpoint file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for the experiment')
    parser.add_argument('--val_subset', type=int, default=None,
                        help='Validation subset to use, in [1000, 3000, 5000, 10000]')
    parser.add_argument('--run_forward', action="store_true", help='Set to run original forward with no dropping')
    parser.add_argument('--run_random_drop', action="store_true", help='Set to run random dropping')
    parser.add_argument('--run_heuristic_drop', action="store_true", help='Set to run heuristic dropping')
    # checkpoint_acc0.75.pth

    exp_args = parser.parse_args()
    return exp_args


if __name__ == '__main__':

    # Example arguments
    exp_args = argparse.Namespace(
        model_dir="ne8_mr8",
        checkpoint="best_checkpoint.pth",
        batch_size=1000,
        val_subset=None,
        run_forward=True,
        run_random_drop=True,
        run_heuristic_drop=True,
    )

    # if args.pkl exists, load that as args
    # -> this would be the case with the latest vmap_fixed_runs
    # elif jobs/main_args.json exists, then load that as args
    # -> this would be the case with the dud vmap runs
    model_args_pkl_file = os.path.join(exp_args.model_dir, "args.pkl")
    model_args_json_file = os.path.join(exp_args.model_dir, "job", "main_args.json")
    if os.path.exists(model_args_pkl_file):
        with open(model_args_pkl_file, "rb") as f:
            model_args = pkl.load(f)
        assert isinstance(model_args, argparse.Namespace)
    elif os.path.exists(model_args_json_file):
        with open(model_args_json_file, "r") as f:
            model_args = json.load(f)
        model_args = argparse.Namespace(**model_args)
    else:
        raise ValueError("No model args file found")

    # Replace some args
    model_args.batch_size = exp_args.batch_size
    model_args.val_subset = exp_args.val_subset
    print(f"Run setting :: NE={model_args.num_experts}, MR={model_args.moe_mlp_ratio}")
    model_args.run_forward = exp_args.run_forward
    model_args.run_random_drop = exp_args.run_random_drop
    model_args.run_heuristic_drop = exp_args.run_heuristic_drop
    ### END: Args

    ### BEGIN: Model
    model_checkpoint_file = os.path.join(exp_args.model_dir, exp_args.checkpoint)
    assert os.path.exists(model_checkpoint_file)
    model_args.resume = model_checkpoint_file
    ### END: Model

    ### BEGIN: NEGATE vmap
    model_args.vmap_model = False
    ### BEGIN: NEGATE vmap

    model_args.pin_mem = True
    model_args.num_workers = 16
    results = main(model_args)

    # Save results
    save_name = f"{exp_args.model_dir.split('/')[-1]}_random_trials.pkl"
    pkl.dump(results, open(save_name, "wb"))
    
    for k,v in results.items():
        print(f"Results for {k}")
        if k == "random":
            for kk,vv in v.items():
                mean_acc = np.mean([x['acc1'] for x in vv])
                stddev_acc = np.std([x['acc1'] for x in vv], ddof=1)
                print(f"  - {kk}: {mean_acc:.2f}% +/- {stddev_acc:.2f}")
        else:
            for kk,vv in v.items():
                print(f"  - {kk}: {vv['acc1']:.2f}%")
        print("\n")

