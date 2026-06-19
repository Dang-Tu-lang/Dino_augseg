import torch
import torch.nn.functional as F

def test_normalize():
    zeros = torch.zeros(2, 3, 256)
    normed = F.normalize(zeros, p=2, dim=-1)
    print("Zeros norm before:", zeros.norm(dim=-1))
    print("Zeros norm after:", normed.norm(dim=-1))
    
    # K-means test
    print("Normed is exactly 0:", (normed == 0).all().item())

test_normalize()
