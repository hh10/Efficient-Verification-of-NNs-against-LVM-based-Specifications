import torch
import torch.nn as nn
import random

from verifiers_utils import get_verifier_reshape_op, verify_segment, Status

# set random seed for reproducibility.
seed = 422
random.seed(seed)
torch.manual_seed(seed)


def segment_verification(ndims, device, i):
    num_classes, ldims, bias = 4, 16, False
    in_dims = ndims*ndims*ldims
    Reshape = get_verifier_reshape_op()
    model_layers = [Reshape((1, ldims, ndims, ndims)),  # will reshape latent space line to 2d for conv operations
                    ## Note: can run ConvTranspose2d by adding its implementation patch for verinet provided in the codebase verinet_patches/*
                    # nn.ConvTranspose2d(ldims, ldims, 1, 1),
                    ## Note: can run LeakyReLU by adding its implementation patch for verinet provided in the codebase verinet_patches/*
                    # nn.LeakyReLU(1e-2),
                    nn.Conv2d(ldims, ldims, 1, 1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(in_dims, in_dims, bias=bias),
                    nn.ReLU(),
                    nn.Linear(in_dims, num_classes, bias=bias)]

    rand_x = torch.rand((2, in_dims), dtype=torch.double)  # 2 pts in in_dim space
    true_class = 1
    print("")
    print("Verifying for input no.:", i)
    status, _, _, ver_time, _ = verify_segment(model_layers, rand_x, true_class, num_classes, 30, device, sanity_checks=True)
    print(f'Verified {status.name} in duration {ver_time:.2f}s')
    return status


if __name__ == '__main__':
    print("Example runs for locally verifying line segment input with custom VeriNetNN:")
    device = torch.device("cpu")  # cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, "will be used for verification.")
    
    results = {Status.Safe: 0, Status.Unsafe: 0, Status.Undecided: 0, Status.Underflow: 0}
    for i in range(10):
        status = segment_verification(random.randrange(4, 6), device, i)
        results[status] += 1
    print(results)
