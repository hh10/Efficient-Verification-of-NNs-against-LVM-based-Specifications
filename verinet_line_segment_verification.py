import numpy as np
import torch
import torch.nn as nn
#import gurobipy as grb
from random import randrange
import time
import copy

from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode
from verinet.neural_networks.custom_layers import Reshape
from verinet.verification.objective import Objective
from verinet.verification.verinet import VeriNet
from verinet.verification.verifier_util import Status

torch.manual_seed(422)


def merge_linear_neighbours(in_layers: list, sanity_check: bool = True) -> list:
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
    rand_x = torch.rand((1, 1)).double()
    with torch.no_grad():
        rand_x1 = nn.Sequential(*in_layers)(rand_x)
        rand_x2 = nn.Sequential(*out_layers)(rand_x)
    assert torch.max(rand_x1-rand_x2) < 1e-6, print(torch.max(rand_x1-rand_x2), rand_x1, rand_x2)
    return out_layers


def setup_bias(layers: list) -> list:
    rand_x = torch.rand((1, 1)).double()
    with torch.no_grad():
        rand_x1 = nn.Sequential(*layers)(rand_x)

    for i, ml in enumerate(layers):
        if isinstance(ml, nn.Linear) and ml.bias is None:
            layers[i] = nn.Linear(ml.in_features, ml.out_features).double()
            layers[i].weight = ml.weight
            layers[i].bias.data.fill_(0.)

    with torch.no_grad():
        rand_x2 = nn.Sequential(*layers)(rand_x)
    assert torch.max(rand_x1-rand_x2) < 1e-6, print(torch.max(rand_x1-rand_x2), rand_x1, rand_x2)
    return layers


def nnLayers_to_VeriNetNNNodes(layers: list) -> list:
    nodes = [VeriNetNNNode(idx=0, op=nn.Identity(), connections_from=None, connections_to=[1])]
    for li, layer in enumerate(layers):
        if any([isinstance(layer, allowed_layer) for allowed_layer in [nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Conv2d, nn.ConvTranspose2d, Reshape, nn.Flatten]]):
            node = VeriNetNNNode(idx=li+1, op=layer, connections_from=[li], connections_to=[li+2])
        else:
            raise NotImplementedError(f'VeriNetNNNode for {layer} not implemented')        
        nodes.append(node)
    nodes.append(VeriNetNNNode(idx=len(layers)+1, op=nn.Identity(), connections_from=[len(layers)], connections_to=None))
    return nodes   


def augment_network_for_ls(pts):
    ndims = pts.shape[1]
    ls_enc_layer1 = nn.Linear(1, ndims, bias=None)
    ls_enc_layer1.weight.data = torch.FloatTensor([1]*ndims).unsqueeze(1)
    ls_enc_layer2 = nn.Linear(ndims, ndims)
    ls_enc_layer2.weight.data = torch.diag(pts[1, :] - pts[0, :])
    ls_enc_layer2.bias.data = pts[0, :]
    return [ls_enc_layer1, ls_enc_layer2]


def get_VerinetNN_solver(layers, device, sanity_checks=False):
    layers = [ml.double().to(device) for ml in layers]
    # merge neighbouring Linear layers to better verification results 
    merged_ver_layers = merge_linear_neighbours(layers, sanity_check=sanity_checks)
    # bias False is not expected in Verinet, so set bias to 0 where bias = False for Linear layers
    merged_ver_layers = setup_bias(merged_ver_layers)
    nodes = nnLayers_to_VeriNetNNNodes(merged_ver_layers)
    return VeriNetNN(nodes), VeriNet(use_gpu=True, max_procs=None)


def verinet_verify_line_segment(mlayers, pts, corr_class, num_classes, timeout_s, sanity_checks=False, device="cpu"):
    """ Verifies line segment between pts[0], pts[1], i.e., pts[0] + alpha * (pts[1]-pts[0]) """
    input_bounds = np.zeros((1, 2), dtype=np.float32)
    input_bounds[:, 1] = 1  # bounds for alpha for the line segment

    # adding layers for constructing line segment between pts
    ls_enc_layers = augment_network_for_ls(pts)
    all_ver_layers = ls_enc_layers + copy.deepcopy(mlayers)

    model, solver = get_VerinetNN_solver(all_ver_layers, device, sanity_checks)
    objective = Objective(input_bounds, output_size=num_classes, model=model)
    out_vars = objective.output_vars
    for j in range(objective.output_size):
        if j != corr_class:
            objective.add_constraints(out_vars[j] <= out_vars[corr_class])
    # reported outputs after modifying opensource VeriNet codebase
    counterex_using_PGD, bounds_summary, mem_usage_bytes = None, None, None
    start_time = time.time()
    verifier_output = solver.verify(objective=objective, timeout=timeout_s)
    ver_time = time.time() - start_time
    if sanity_checks:
        print(f"Verification results: {solver.status}")
        print(f"Branches explored: {solver.branches_explored}")
        print(f"Maximum depth reached: {solver.max_depth}")
        print(f"Memory: {solver.branches_explored}")
        if type(verifier_output) is list:
            status, counterex_using_PGD, jsip_bounds, mem_usage_bytes = verifier_output  # can modify VeriNet codebase to return all these values
            print(f"cex_using_PGD: {counterex_using_PGD}")
            if jsip_bounds is not None:
                print(f"jsip_bounds: {jsip_bounds.shape}")
        else:
            status = verifier_output

    # Store the counter example if Unsafe. Status enum is defined in src.algorithm.verinet_util
    ceg, ceg_alpha = None, None
    if status == Status.Unsafe:
        ceg_alpha = torch.DoubleTensor(solver.counter_example).to(device)
        # ceg above is the alpha, generate a latent space counterexample from it, i.e., the actual point on the segment
        ceg = nn.Sequential(*ls_enc_layers)(ceg_alpha)
        if sanity_checks:
            counterex_sanity_checks(ceg, ceg_alpha, all_ver_layers, pts, corr_class)

    if sanity_checks:
        # check that the entire line segment joining the pts is being verified against
        # by checking the points corresponding to the bounds are indeed pts
        with torch.no_grad():
            ls_endpoints = nn.Sequential(*ls_enc_layers)(torch.DoubleTensor(input_bounds).t())
        assert torch.max(ls_endpoints - pts) < 1e-3, print(ls_endpoints, pts)
    solver.cleanup()
    del solver
    bounds_summary = None
    if type(verifier_output) is list and jsip_bounds is not None:
        bounds_diff = jsip_bounds[:, 1] - jsip_bounds[:, 0]
        bounds_summary = [torch.min(bounds_diff).item(), torch.mean(bounds_diff).item(), torch.max(bounds_diff).item()]
    return status, ceg, counterex_using_PGD, ver_time, ceg_alpha, bounds_summary, mem_usage_bytes


def assert_pt_on_line_segment(pt, pts):
    if torch.all(pt == pts[0]) or torch.all(pt == pts[1]):
        print("Counter-example an endpoint")
        return
    v21 = pts[1, :] - pts[0, :]
    vpt1 = torch.DoubleTensor(pt) - pts[0, :]
    # for i in range(pt.shape[0]-1):
    i = 0
    assert abs(vpt1[i+1]/vpt1[i] - v21[i+1]/v21[i]) < 1e-3, print(vpt1[i+1]/vpt1[i], v21[i+1]/v21[i])


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


def test_verinet_ls(ndims, device, i):
    num_classes, ldims, bias = 4, 16, False
    in_dims = ndims*ndims*ldims
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
    res, ceg, _, ver_time, _, _, _ = verinet_verify_line_segment(model_layers, rand_x, true_class, num_classes, 30, True)
    # todo(hh): add hist of ver_time
    print(f'Verified {res} in duration {ver_time/60:.2f}m')
    return res, ceg


if __name__ == '__main__':
    # Get the "Academic license" print from gurobi at the beginning
    #grb.Model()
    print("\nExamples for locally verifying line segment input with custom VeriNetNN:")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, "will be used for verification.")
    
    results = {Status.Safe: 0, Status.Unsafe: 0, Status.Undecided: 0, Status.Underflow: 0}
    for i in range(10):
        res, ceg = test_verinet_ls(randrange(4, 6), device, i)
        results[res] += 1
    print(results)
