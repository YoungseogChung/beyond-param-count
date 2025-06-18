import tqdm
import numpy as np
import torch

from evaluation_scripts.top_k_accuracy import top_k_accuracy


def get_accuracy_on_loader_with_model_function(model_call_fn, dataloader, device, 
                                               evaluate_full_model_full_data_tiny_classes,
                                               convert_tiny_data_full_model=None):
    logits_list = []
    pred_class_list = []
    true_y_list = []
    topk_cand = [3, 5, 10]
    topk_acc = {}
    for cur_x, cur_y in tqdm.tqdm(dataloader):
        cur_x = cur_x.to(device)
        cur_y = cur_y.detach().cpu().numpy()

        logits_batch = model_call_fn(x=cur_x).detach().cpu().numpy()
        assert len(logits_batch.shape) == 2
        assert logits_batch.shape[0] == cur_x.shape[0]
        pred_class_batch = logits_batch.argmax(axis=1)

        logits_list.append(logits_batch)
        pred_class_list.append(pred_class_batch)
        true_y_list.append(cur_y)

    logits = np.concatenate(logits_list, axis=0)
    pred_class = np.concatenate(pred_class_list, axis=0)
    true_y = np.concatenate(true_y_list, axis=0)


    if evaluate_full_model_full_data_tiny_classes:
        # evaluate full model on full data, but only on tiny classes
        assert isinstance(convert_tiny_data_full_model, dict)
        is_tiny_class = np.isin(true_y, convert_tiny_data_full_model["full_class_idx_list"])
        logits = logits[is_tiny_class]
        pred_class =  pred_class[is_tiny_class]
        true_y = true_y[is_tiny_class]
    elif convert_tiny_data_full_model is not None:
        assert isinstance(convert_tiny_data_full_model, dict)
        # data is tiny, model is full
        # convert_tiny_data_full_model is a dict with keys tiny_class_idx_list, full_class_idx_list, full_class_name_list
        # take only the classes for which full_imagenet also has data for 
        tiny_class_is_in_full_data_classes = np.isin(true_y, convert_tiny_data_full_model["tiny_class_idx_list"])
        logits = logits[tiny_class_is_in_full_data_classes]
        pred_class =  pred_class[tiny_class_is_in_full_data_classes]
        true_y_with_tiny_classes = true_y[tiny_class_is_in_full_data_classes]
        # just have to translate tiny_class numbers to full_class numbers
        mapping = {
            tiny_idx : full_idx 
            for (tiny_idx, full_idx) in 
            zip(convert_tiny_data_full_model["tiny_class_idx_list"], 
                convert_tiny_data_full_model["full_class_idx_list"])
        }
        translated_true_y = true_y_with_tiny_classes.copy()
        for k, v in mapping.items():
            translated_true_y[true_y_with_tiny_classes == k] = v
        true_y = translated_true_y

    top1_acc = np.mean((pred_class == true_y).astype(float))
    for k in topk_cand:
        cur_topk_acc = top_k_accuracy(logits, true_y, k=k)
        topk_acc[k] = cur_topk_acc

    return top1_acc, topk_acc


# This is really for just validation during trainig
# :returns both loss and accuracy
def validate(model, val_loader, criterion_sum, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # inputs = batch_data["pixel_values"]
            # targets = batch_data["label"]
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = model(inputs)
            logits = model.predict(x=inputs)
            loss = criterion_sum(logits, targets)
            _, predicted = logits.max(1)
            total += targets.size(0)
            val_loss += loss.item()
            correct += predicted.eq(targets).sum().item()

            # del inputs, targets, outputs, loss, predicted
            # gc.collect()

    val_loss /= total
    accuracy = 100.0 * correct / total
    return val_loss, accuracy