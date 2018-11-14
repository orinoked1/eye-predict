import torch

use_cuda = torch.cuda.is_available()

print(use_cuda)