
import os
import argparse
import yaml

import numpy as np
import pickle as pkl
import torch
import torch.nn as nn

from timm import utils
from timm.data import (
    create_dataset, 
    create_loader, 
    resolve_data_config
)
from timm.models import (
    create_model, 
    safe_model_name, 
    resume_checkpoint, 
    load_checkpoint, 
    model_parameters
)
# My own code
from timm.smoe_files.validate_for_drop_experiment import validate

def parse_exp_args():
    parser = argparse.ArgumentParser(description='Run drop experiment')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the model directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for the experiment')
    parser.add_argument('--drop_type', type=str, choices=['random', 'heuristic'],
                        required=True, help='Type of dropping to perform')
    parser.add_argument('--checkpoint', type=str, default="model_best.pth.tar", 
                        help='Path to the checkpoint file')

    exp_args = parser.parse_args()
    return exp_args


def main():
    ####################
    collect_results = {}
    ####################
    for ne in [2, 8, 16, 64, 128]:
        print(f"Running for {ne} experts")
        ###############
        ne_results = {}
        ###############
        mr = 64 / ne
        if not mr < 1:
            mr = int(mr)
            
        exp_args = argparse.Namespace(
            # for the cifar100 results already in the paper:
            # model_dir=f'/home/scratch/youngsec/rs/moe/git_packages/astroformer-smoe/pytorch-image-models/smoe_a1/ne{ne}',
            # for the final rerun of astroformer
            model_dir=f'/zfsauton/project/public/ysc/smoe-astro/wo_skip_redo/ne{ne}_mr{mr}',
            checkpoint='model_best.pth.tar',
            batch_size=500,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model args from model_dir
        args_fpath = os.path.join(exp_args.model_dir, 'args.yaml')
        with open(args_fpath, 'r') as stream:
            data = yaml.safe_load(stream)
        args = argparse.Namespace(**data)
        assert args.num_experts == ne
        args.model_kwargs["smoe_params"] = {
            "num_experts": args.num_experts,
            "moe_ratio": args.moe_ratio,
        }
        args.model_kwargs["img_size"] = args.img_size
        args.resume = os.path.join(exp_args.model_dir, exp_args.checkpoint)

        ###
        args.checkpoint = os.path.join(exp_args.model_dir, exp_args.checkpoint)
        args.use_ema = False
        args.device = device
        args.reparam = False
        args.use_train_size = False
        args.test_pool = None
        args.aot_autograd = False
        args.num_gpu = 1
        args.split = args.val_split
        args.tf_preprocessing = False
        args.valid_labels = ''
        args.real_labels = ''
        args.batch_size = exp_args.batch_size
        args.log_freq = 10

        args.amp = False
        ###

        ### BEGIN: Original forward
        args.drop_type = None
        args.num_heuristic_drop = None
        results = validate(args)
        ne_results["drop_0"] = results
        torch.cuda.empty_cache()
        ### END: Original forward
            
        drop_cand_list = list(set([ne // 2, 3 * ne // 4, 7 * ne // 8]))
        if 0 in drop_cand_list:
            drop_cand_list.remove(0)
        drop_cand_list = np.array(sorted(drop_cand_list))

        print(drop_cand_list)
        for num_drop in drop_cand_list:
            print(f"Running for {num_drop} drops")
            ### BEGIN: Heuristic drop
            args.drop_type = "heuristic"
            args.num_heuristic_drop = num_drop
            alg_results = validate(args)
            assert f"heuristic_drop_{num_drop}" not in ne_results
            ne_results[f"heuristic_drop_{num_drop}"] = alg_results
            torch.cuda.empty_cache()
            print(f"Results for {num_drop} drops; heuristic")
            print(alg_results['top1'])
            ### END: Heuristic drop

            ### BEGIN: Random drop
            rand_res_list = []
            for rand_i in range(10):
                torch.manual_seed(rand_i)
                np.random.seed(rand_i)
                args.drop_type = "random"
                args.num_heuristic_drop = num_drop
                results = validate(args)
                assert f"random_drop_{num_drop}" not in ne_results
                # ne_results[f"random_drop_{num_drop}"] = results
                torch.cuda.empty_cache()
                print(f"Results for {num_drop} drops; random")
                ne_results[f"random_drop_{num_drop}_{rand_i}"] = results
                print(results['top1'])
            ### END: Random drop
        collect_results[ne] = ne_results

        # Saving results
        save_fname = f"wo_skip_redo_astro_drop_results_ne{ne}.pkl"
        pkl.dump(collect_results, open(save_fname, 'wb'))


if __name__ == '__main__':
    main()
    
