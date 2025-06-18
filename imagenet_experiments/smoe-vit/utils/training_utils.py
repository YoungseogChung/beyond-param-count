import os
import sys
import torch

def save_command(save_dir, save_file_name="command.txt"):
    full_command = ' '.join(sys.argv)
    with open(f"{save_dir}/{save_file_name}", "w") as f:
        f.write(full_command)
    print(f"Full command used: {full_command}")


def save_model_and_optimizer(model, optimizer, model_save_file_name, optimizer_save_file_name):
    if os.path.exists(model_save_file_name):
        print(f"Over-writing model saved")
    if os.path.exists(optimizer_save_file_name):
        print(f"Over-writing optimizer saved")
    torch.save(model.state_dict(), model_save_file_name)
    torch.save(optimizer.state_dict(), optimizer_save_file_name)
    print(f"Model and optimizer saved to {model_save_file_name} and {optimizer_save_file_name}")


def find_file_with_string(search_dir, search_str):
    matches = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if search_str in file:
                matches.append(os.path.join(root, file))

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple files found with the string '{search_str}':")
        for match in matches:
            print(match)
    else:
        print(f"No files found with the string '{search_str}'")



# BEGIN: here's another way of loading weights - from checkpoint safetensors
# import safetensors
# safetensor_path = "/zfsauton/project/public/ysc/tiny-smoe-vit/ne_2-ms_256-model_tiny-lr_0.001/results/model_29250.safetensors"
# weights = safetensors.torch.load_file(safetensor_path)
# model.load_state_dict(weights)
# END