import torch
from src.enums.hyperparameters import NB_CLASSES, L_MAX, BATCH_SIZE
from src.training.random_model import RandomModel
import torch.nn as nn

from src.utils.metrics import accuracy_fn


def training(epochs, random_model, x, mask, y, loss_fn, optimizer):
    all_accuracy, all_losses = [], []
    for epoch in range(epochs):
        # Training
        random_model.train()
        # 1. Forward pass
        # y_logits size: (N_BATCH , L_MAX, NB_CLASSES)
        y_logits = random_model(x)
        # 2. Reshape the output to be of size (N_BATCH * L_MAX, NB_CLASSES)
        y_pred = y_logits[mask].view(-1, NB_CLASSES)
        # Reshape y_true that should be of size (N_BATCH*L_MAX)
        loss = loss_fn(y_pred,
                       y[mask].view(-1))
        all_losses.append(loss.item())
        acc = accuracy_fn(y_true=y[mask].view(-1),
                          y_pred=y_pred)
        all_accuracy.append(acc)
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backwards
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}")



if __name__ == "__main__":
    x = torch.randn(BATCH_SIZE, L_MAX, 4)
    mask = torch.randint(0, 2, (BATCH_SIZE, L_MAX)) # Mask to ignore padding positions
    y = torch.randint(0, 10, (BATCH_SIZE, L_MAX))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    random_model = RandomModel(NB_CLASSES).to(device)
    optimizer = torch.optim.Adam(params=random_model.parameters(),
                                 lr=0.01)
    EPOCHS = 1000
    # Do the training process
    training(EPOCHS, random_model, x, mask, y, loss_fn, optimizer)
