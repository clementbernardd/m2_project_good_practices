import einops
from torch.nn import BatchNorm1d, LazyBatchNorm1d, InstanceNorm1d
import torch.nn as nn

class RandomModel(nn.Module):

  def __init__(self, nb_classes, hidden_size = 128):
    super().__init__()
    self.layer = nn.Conv1d(in_channels=4, out_channels=hidden_size, kernel_size=15, padding='same')
    self.activation = nn.Sequential(BatchNorm1d(hidden_size), nn.ELU(), nn.Dropout(p=0.2))
    self.classifier = nn.Linear(hidden_size, nb_classes)

  def forward(self, x):
    batch_size = len(x)
    x = einops.rearrange(x, "b l f -> b f l")
    z = self.layer(x)
    z = self.activation(z)
    z = einops.rearrange(z, 'b f l -> (b l) f')
    z = nn.Softmax()(self.classifier(z))
    z = einops.rearrange(z, "(b l) c -> b l c", b = batch_size)
    return z