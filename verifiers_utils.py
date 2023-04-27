import torch
from torch import nn

import numpy as np
from enum import Enum
import copy
import time
import random


class Status(Enum):
    Safe = 1
    Unsafe = 2
    Undecided = 3
    Underflow = 4
    Mem_error = 5
    Fp_precision_error = 6


def segment_encoding_layers(pts):
    ndims = pts.shape[1]
    ls_enc_layer1 = nn.Linear(1, ndims, bias=None)
    ls_enc_layer1.weight.data = torch.FloatTensor([1]*ndims).unsqueeze(1)
    ls_enc_layer2 = nn.Linear(ndims, ndims)
    ls_enc_layer2.weight.data = torch.diag(pts[1, :] - pts[0, :])
    ls_enc_layer2.bias.data = pts[0, :]
    return [ls_enc_layer1, ls_enc_layer2]


class VerificationBackend(Enum):
    OpensourceVerinet = "opensource_verinet"
    # ProprietaryVerinet = "proprietary_verinet"  # contact authors for this version+its patches and uncomment
    DEFAULT = OpensourceVerinet  # change here to use a different supported verifier
# add more verification backends as implementing their support and can change the DEFAULT_VERIFIER constant to use a specific one


def get_verifier_solver(layers, device, sanity_checks, verifierBackend=VerificationBackend.DEFAULT):
    layers = [copy.deepcopy(ml).double().to(device) for ml in layers]
    # merge neighbouring Linear layers to better verification results 
    merged_ver_layers = merge_linear_neighbours(layers, device, sanity_check=sanity_checks)
    # bias False is not expected in Verinet, so set bias to 0 where bias = False for Linear layers
    merged_ver_layers = setup_bias(merged_ver_layers, device, sanity_check=sanity_checks)

    if verifierBackend == VerificationBackend.OpensourceVerinet:
        from verinet.neural_networks.verinet_nn import VeriNetNN
        from verinet.verification.verinet import VeriNet
        nodes = nnLayers_to_VeriNetNNNodes(merged_ver_layers)
        return VeriNetNN(nodes), VeriNet(use_gpu=device != torch.device("cpu"), max_procs=None)
    raise NotImplementedError(f"Verifier backend {verifierBackend.name} not supported.")


def get_classification_objective(input_bounds, corr_class, num_classes, verifierModel, verifierBackend=VerificationBackend.DEFAULT):
    if verifierBackend == VerificationBackend.OpensourceVerinet:
        from verinet.verification.objective import Objective
    else:
        raise NotImplementedError(f"Objective for Verifier backend {verifierBackend.name} not known.")
    
    objective = Objective(input_bounds, output_size=num_classes, model=verifierModel)
    out_vars = objective.output_vars
    for j in range(objective.output_size):
        if j != corr_class:
            objective.add_constraints(out_vars[j] <= out_vars[corr_class])
    return objective


def verify_segment(mlayers, pts, corr_class, num_classes, timeout_s, device, sanity_checks=False, verifierBackend=VerificationBackend.DEFAULT):
    input_bounds = np.zeros((1, 2), dtype=np.float32)
    input_bounds[:, 1] = 1  # bounds for alpha for the line segment

    # adding layers for constructing line segment between pts
    ls_enc_layers = segment_encoding_layers(pts)
    layers = ls_enc_layers + mlayers

    # prepare the verifier wrapper, solver and objective
    vmodel, solver = get_verifier_solver(layers, device, sanity_checks, verifierBackend)
    objective = get_classification_objective(input_bounds, corr_class, num_classes, vmodel, verifierBackend)
    status, ceg_alpha, ceg_using_PGD, ver_time = verify(solver, objective, timeout_s, device, sanity_checks)
    ceg = None
    if ceg_alpha is not None:
        # ceg above is the alpha, generate a latent space counterexample from it, i.e., the actual point on the segment
        ceg = nn.Sequential(*ls_enc_layers)(ceg_alpha)
        if sanity_checks:
            counterex_sanity_checks(ceg, ceg_alpha, layers, pts, corr_class)

    if sanity_checks:
        # check that the entire line segment joining the pts is being verified against
        # by checking the points corresponding to the bounds are indeed pts
        with torch.no_grad():
            ls_endpoints = nn.Sequential(*ls_enc_layers).to("cpu")(torch.DoubleTensor(input_bounds).t())
        assert torch.max(ls_endpoints - pts) < 1e-3, print(ls_endpoints, pts)
    return status, ceg, ceg_using_PGD, ver_time, ceg_alpha


def verify(solver, objective, timeout_s, device, sanity_checks=False):
    # reported outputs after modifying opensource VeriNet codebase
    start_time = time.time()
    verifier_output = solver.verify(objective=objective, timeout=timeout_s)
    ver_time = time.time() - start_time
    if sanity_checks:
        print(f"Verification results: {solver.status}, Branches explored: {solver.branches_explored}, Maximum depth reached: {solver.max_depth}, Memory: {solver.branches_explored}")
    else:
        print(f"Verification output: {verifier_output}")

    if type(verifier_output) is list:
        status, ceg_using_PGD, jsip_bounds, mem_usage_bytes = verifier_output  # can modify VeriNet codebase to return all these values
        if sanity_checks:
            print(f"Cex_using_PGD: {ceg_using_PGD}")
        if jsip_bounds is not None:
            print(f"jsip_bounds: {jsip_bounds.shape}")
            bounds_diff = jsip_bounds[:, 1] - jsip_bounds[:, 0]
            bounds_summary = [torch.min(bounds_diff).item(), torch.mean(bounds_diff).item(), torch.max(bounds_diff).item()]
    else:
        status, ceg_using_PGD, bounds_summary, mem_usage_bytes = verifier_output, None, None, None
    status = Status(status.value)

    ceg = torch.DoubleTensor(solver.counter_example).to(device) if status == Status.Unsafe else None
    solver.cleanup()
    del solver
    return status, ceg, ceg_using_PGD, ver_time  # , bounds_summary, mem_usage_bytes


def get_verifier_reshape_op(verifierBackend: VerificationBackend = VerificationBackend.OpensourceVerinet):
    if verifierBackend == VerificationBackend.OpensourceVerinet:
        from verinet.neural_networks.custom_layers import Reshape
        return Reshape
    raise NotImplementedError(f"Verifier backend {verifierBackend.name} Reshape operation not known/supported.")


# utils
def merge_linear_neighbours(in_layers: list, device, sanity_check: bool = True) -> list:
    out_layers, linear_layers = [], []
    for i, ml in enumerate(in_layers):
        if isinstance(ml, nn.Linear) and ml.bias is None and i < len(in_layers)-1:
            # keep accumulating linear layers with 0 bias for merging (unless last layer)
            linear_layers.append(ml)
        else:
            if len(linear_layers) == 0:
                out_layers.append(ml)
                continue
            if isinstance(ml, nn.Linear):
                linear_layers.append(ml)
            # merging accumulated linear layers
            llayer_bias = linear_layers[-1].bias is not None
            mlayer = nn.Linear(linear_layers[0].in_features, linear_layers[-1].out_features, bias=llayer_bias).double()
            mlayer_weights = linear_layers[0].weight.data
            for j in range(1, len(linear_layers)):
                mlayer_weights = torch.mm(linear_layers[j].weight.data, mlayer_weights)
            mlayer.weight.data = nn.Parameter(mlayer_weights)

            if llayer_bias:
                mlayer.bias.data = linear_layers[-1].bias.data

            out_layers.append(mlayer)
            linear_layers = []
            if not isinstance(ml, nn.Linear):
                out_layers.append(ml)
    # sanity check
    if sanity_check:
        rand_x = torch.rand((1, 1)).double().to(device)
        with torch.no_grad():
            rand_x1 = nn.Sequential(*in_layers)(rand_x)
            rand_x2 = nn.Sequential(*out_layers)(rand_x)
        assert torch.max(rand_x1-rand_x2) < 1e-6, print(torch.max(rand_x1-rand_x2), rand_x1, rand_x2)
    return out_layers


def setup_bias(layers: list, device, sanity_check: bool = True) -> list:
    if sanity_check:
        rand_x = torch.rand((1, 1)).double().to(device)
        with torch.no_grad():
            rand_x1 = nn.Sequential(*layers)(rand_x)

    for i, ml in enumerate(layers):
        if isinstance(ml, nn.Linear) and ml.bias is None:
            layers[i] = nn.Linear(ml.in_features, ml.out_features).double().to(device)
            layers[i].weight = ml.weight
            layers[i].bias.data.fill_(0.)

    if sanity_check:
        with torch.no_grad():
            rand_x2 = nn.Sequential(*layers)(rand_x)
        assert torch.max(rand_x1-rand_x2) < 1e-6, print(torch.max(rand_x1-rand_x2), rand_x1, rand_x2)
    return layers


def nnLayers_to_VeriNetNNNodes(layers: list) -> list:
    from verinet.neural_networks.verinet_nn import VeriNetNNNode
    from verinet.neural_networks.custom_layers import Reshape

    nodes = [VeriNetNNNode(idx=0, op=nn.Identity(), connections_from=None, connections_to=[1])]
    for li, layer in enumerate(layers):
        if any([isinstance(layer, allowed_layer) for allowed_layer in [nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Conv2d, nn.ConvTranspose2d, Reshape, nn.Flatten]]):
            node = VeriNetNNNode(idx=li+1, op=layer, connections_from=[li], connections_to=[li+2])
        else:
            raise NotImplementedError(f'VeriNetNNNode for {layer} not implemented')        
        nodes.append(node)
    nodes.append(VeriNetNNNode(idx=len(layers)+1, op=nn.Identity(), connections_from=[len(layers)], connections_to=None))
    return nodes


def counterex_sanity_checks(ceg, ceg_alpha, layers, pts, corr_class):
    with torch.no_grad():
        out_logits = nn.Sequential(*layers)(ceg_alpha)
        _, pred_class = torch.max(out_logits, axis=1)
    alpha_ceg_false = pred_class == corr_class
    print(f"Alpha counterexample is true/good?: {not alpha_ceg_false.item()} (True class: {corr_class}, Predicted class: {pred_class.item()})")
    # check that the counter example lies on the line-segment
    ceg_hat = pts[0, :] + ceg_alpha*(pts[1, :] - pts[0, :])
    assert torch.max(ceg-ceg_hat) < 1e-4, print(torch.max(ceg-ceg_hat), ceg, ceg_hat, ceg_alpha)
    with torch.no_grad():
        out_logits = nn.Sequential(*layers[2:]).double()(ceg.double())
        _, pred_class = torch.max(out_logits, axis=1)
    img_ceg_false = pred_class == corr_class
    print(f"Image counterexample is true/good?: {not img_ceg_false.item()} (True class: {corr_class}, Predicted class: {pred_class.item()})")
    assert alpha_ceg_false == img_ceg_false, print(alpha_ceg_false.item(), img_ceg_false.item())


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

    ## add verifiers
    # import sys
    # sys.path.append(f"verifiers/{VerificationBackend.DEFAULT.value}")

    # set random seed and device for reproducibility.
    seed = 422
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")  # cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, "will be used for verification.")

    results = {Status.Safe: 0, Status.Unsafe: 0, Status.Undecided: 0, Status.Underflow: 0}
    for i in range(10):
        status = segment_verification(random.randrange(4, 6), device, i)
        results[status] += 1
    print(results)
