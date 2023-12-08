import torch

def accuracy_fn(y_true, y_pred):
    pred_classes = torch.argmax(y_pred, axis = -1)
    correct = torch.eq(y_true, pred_classes).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc