import os
import json
import argparse
from datetime import datetime

import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import (
    set_seed,
    print_args,
)
from soft_moe.soft_moe_utils import make_phi
from soft_moe.soft_moe_model import (
    SoftMoeModel,
    get_model_predictions,
)
from models_and_data.mnist import construct_mnist_components, make_dummy_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="soft or baseline")
    parser.add_argument("--setting", type=str, default="mnist")
    # parser.add_argument('--setting', type=str, default='synthetic_1d')

    # training arguments
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--optimizer", type=str, default="sgd")

    # data arguments
    parser.add_argument(
        "--flip_pixels",
        action="store_true",
        help="whether to flip the pixels of the input",
    )
    parser.add_argument("--patch_size", type=int, default=None)

    # MoE arguments
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--routing_type", type=str, default="soft_moe")
    parser.add_argument("--phi_init", type=str, default="uniform")
    parser.add_argument("--num_slots", type=int, default=1)  # soft-moe slots per expert
    parser.add_argument(
        "--expert_out_dim",
        type=int,
        default=None,
        help="the dimension that the experts output",
    )
    # MoE variation arguments
    parser.add_argument(
        "--expert_fn_multiplier", type=int, default=1
    )  # multiplier for parameters of mlp_small expert_fn
    parser.add_argument(
        "--uniform_d", action="store_true"
    )  # whether setting D to be uniform
    parser.add_argument(
        "--uniform_c", action="store_true"
    )  # whether setting C to be uniform
    parser.add_argument(
        "--freeze_phi", action="store_true"
    )  # whether to freeze phi training

    # encoder
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument(
        "--encoder_out_dim",
        type=int,
        default=64,
        help="the dimension that the experts see",
    )  # 512

    # decoder
    parser.add_argument("--decoder", type=str, default=None)
    # parser.add_argument("--decoder_dim", type=int, default=64)  # should the same as encoder

    # Dummy patches
    parser.add_argument("--num_dummy", type=int, default=0)
    parser.add_argument("--type_dummy", type=str, default="min")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # device
    parser.add_argument("--gpu", type=str, default="0")

    # save logs
    parser.add_argument("--save_dir", type=str, default="./logs")
    parser.add_argument("--save_interim", action="store_true")
    parser.add_argument(
        "--save_x", action="store_true", help="whether to save the input data"
    )

    # debug flag
    parser.add_argument("--dbg", action="store_true")
    parser.add_argument(
        "--dry", action="store_true", help="whether to run a dry run (no training)"
    )

    # run name
    parser.add_argument("--name", type=str, default=None)

    # whether to use the normalization before passing to MoE layer
    parser.add_argument("--normalize_moe_input", type=bool, default=False)

    args = parser.parse_args()

    return args


def make_run_identifier(args):
    out_str = ""
    if args.name is not None:
        out_str += f"{args.name}"
    else:
        # if name is not given, use the timestamp as name
        out_str += f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    return out_str


def main(args):
    # Set seed
    set_seed(args.seed)
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Get data, encoder and expert functions, and loss function
    ## Note: some additional arguments are set here
    
    if args.setting == "mnist":
        comps = construct_mnist_components(args)
    else:
        raise ValueError(f"Unknown setting {args.setting}")

    # Set run id
    run_id = make_run_identifier(args)
    run_dir = f"{args.save_dir}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    # Save the args
    with open(f"{run_dir}/args.json", "w") as file:
        json.dump(vars(args), file, indent=4)

    # Parse experiment components
    tr_loader = comps["tr_loader"]
    te_loader = comps["te_loader"]
    loss_fn = comps["loss_fn"]
    patch_fn = comps["patch_fn"]
    encoder = comps["encoder"]
    decoder = comps["decoder"]
    expert_fn = comps["expert_fn"]

    if args.dbg:
        breakpoint()

    # Make phi
    phi = make_phi(args, device)
    if args.freeze_phi:
        phi.requires_grad = False
        init_phi = phi.detach().cpu().numpy().copy()

    # Make the experts
    expert_list = [expert_fn().to(device) for _ in range(args.num_experts)]

    # Make SMoE
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
    smoe = smoe.to(device)

    # Make the optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=smoe.parameters(), lr=float(args.lr))
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params=smoe.parameters(), lr=float(args.lr))
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    ### BEGIN: Sanity printing
    print(f"Patching function: {patch_fn}")
    print(f"Encoder: {encoder}")
    print(f"Num Experts: {len(expert_list)}")
    print(f"Expert: {expert_list[0]}")
    num_params_per_expert = sum(
        p.numel() for p in expert_list[0].parameters() if p.requires_grad
    )
    print(f"Num Params per Expert: {num_params_per_expert}")
    print(f"Total Expert Params: {num_params_per_expert * len(expert_list)}")
    print(f"Decoder: {decoder}")

    if args.dry:
        print_args(args)
        return
    ### END: Sanity printing

    # Train logging
    tr_loss_list = []
    tr_acc_list = []

    # Train
    smoe.train()

    # BEGIN: hack -- we don't want to shuffle for now
    assert isinstance(tr_loader.sampler, torch.utils.data.SequentialSampler)
    # END: hack

    # First, iterate over the train set to save it and estimate any distributions
    x_token_list = []
    for batch_data in tr_loader:
        if len(batch_data) == 2:
            X, y = batch_data
            aux_info = None
        elif len(batch_data) == 3:
            X, y, aux_info = batch_data
        # X is of shape (B, C, H, W)
        X = patch_fn(X)
        x_token_list.append(X.detach().cpu().numpy().copy())
    all_x_tokens = np.concatenate(x_token_list, axis=0)  # (N, num_tokens, patch_dim)

    # Now figure out what to do with dummy variables
    dummy_token_fn = make_dummy_tokens(
        type_dummy=args.type_dummy, num_dummy=args.num_dummy, X=all_x_tokens
    )

    # Training
    last_ep_idx = args.num_epochs - 1
    """ BEGIN: One epoch """
    for ep_idx in tqdm.tqdm(range(args.num_epochs)):
        ep_tr_loss_list = []
        pred_class_list = []
        true_class_list = []
        phi_list = []
        Y_tilde_list = []

        """ BEGIN: One batch """
        for batch_data in tr_loader:
            if len(batch_data) == 2:
                X, y = batch_data
                aux_info = None
            elif len(batch_data) == 3:
                X, y, aux_info = batch_data
            X = X.to(device).float()
            y = y.to(device)
            batch_size = X.shape[0]

            Y = smoe.forward(X)
            loss = loss_fn(Y, y)
            ep_tr_loss_list.append(loss.item())

            if args.setting in ["mnist", "cifar10"]:
                # log for classification accuracy
                pred_class = Y.argmax(1)
                pred_class_list.append(pred_class.detach().cpu().numpy().copy())
                true_class_list.append(y.detach().cpu().numpy().copy())

            if ep_idx == 0:
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # checks after each batch step
            if args.freeze_phi:
                assert np.allclose(phi.detach().cpu().numpy(), init_phi)
        """ END: One batch """

        ### BEGIN: save the model after each epoch
        if (hasattr(args, "save_every") and ep_idx % args.save_every == 0) or (ep_idx == last_ep_idx):
            smoe.save(save_dir=run_dir, save_name=f"ep{ep_idx}")
        ### END: save the model after each epoch

        # Append the average loss for the epoch
        tr_loss_list.append(np.mean(ep_tr_loss_list))
        print(f"last loss value {tr_loss_list[-1]:.4f}")
        plt.plot(tr_loss_list)
        plt.title("Loss")
        plt.savefig(f"{run_dir}/loss.png")
        plt.clf()
        torch.save(tr_loss_list, f"{run_dir}/loss.pt")
        if args.setting in ["mnist", "cifar10"]:
            # Get training classification accuracy
            pred_class = np.concatenate(pred_class_list, axis=0)
            true_class = np.concatenate(true_class_list, axis=0)
            acc = np.mean(pred_class == true_class)
            print(f"Accuracy: {acc * 100:.1f}%")
            tr_acc_list.append(acc)

            plt.plot(tr_acc_list)
            plt.ylim(0, 1)
            plt.title("Accuracy")
            plt.savefig(f"{run_dir}/acc.png")
            plt.clf()
            torch.save(tr_acc_list, f"{run_dir}/acc.pt")

        ### print test acc
        pred_class_list = []
        true_class_list = []
        test_loss_list = []
        for batch_data in te_loader:
            if len(batch_data) == 2:
                X, y = batch_data
                aux_info = None
            elif len(batch_data) == 3:
                X, y, aux_info = batch_data
            X = X.to(device).float()
            y = y.to(device)
            
            Y = smoe.forward(X)
            # log for classification accuracy
            pred_class = Y.argmax(1)
            pred_class_list.append(pred_class.detach().cpu().numpy().copy())
            true_class_list.append(y.detach().cpu().numpy().copy())
            loss = loss_fn(Y, y)
            test_loss_list.append(loss.item())

        pred_class = np.concatenate(pred_class_list, axis=0)
        true_class = np.concatenate(true_class_list, axis=0)
        acc = np.mean(pred_class == true_class)
        print(f"Test Loss: {np.mean(test_loss_list):.4f}")
        print(f"Test Accuracy: {acc * 100:.1f}%")
        print("____________________")

        if not getattr(args, "save_interim", False):
            if ep_idx < last_ep_idx:
                continue

        torch.save(tr_loss_list, f"{run_dir}/loss.pt")
        # torch.save(Y_tilde_list, f"{run_dir}/Y_tilde_ep{ep_idx}.pt")
        # torch.save(phi_list, f"{run_dir}/phi_ep{ep_idx}.pt")


    """ END: One epoch """

    # # Saving the final state of everything after all training
    # if args.save_x:
    #     torch.save(y_list, f"{run_dir}/y_list.pt")
    #     torch.save(X_list, f"{run_dir}/X_list.pt")


if __name__ == "__main__":
    # BEGIN: Running a single setting
    args = argparse.Namespace(
        method="soft",
        setting="mnist",
        num_epochs=15,
        lr=0.05,
        batch_size=256,
        optimizer="sgd",
        flip_pixels=False,
        patch_size=14,
        num_experts=-1,  # HACK: make sure I overwrite it
        routing_type="soft_moe",
        phi_init="uniform",
        num_slots=1,
        expert_out_dim=196,
        expert_fn_multiplier=1,
        uniform_d=False,
        uniform_c=False,
        freeze_phi=False,
        encoder=None,
        encoder_out_dim=196,
        decoder="linear",
        num_dummy=0,
        type_dummy="min",
        seed=0,
        gpu="0",
        save_dir="logs/smoe_mnist",
        save_interim=True,
        save_x=False,
        dbg=False,
        dry=False,
        name=None,  # HACK: make sure I overwrite it
    )
    for ne in [1, 2, 3]:
        args.num_experts = ne
        args.name = f"ne_{ne}"
        main(args)
    # END: Running a single setting
