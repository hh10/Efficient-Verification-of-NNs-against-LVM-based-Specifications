import torch
import numpy as np
from verinet import JSIP, ONNXParser, CONFIG
import sys

CONFIG.PRECISION = 32
CONFIG.INPUT_NODE_SPLIT = False
CONFIG.HIDDEN_NODE_SPLIT = False
CONFIG.INDIRECT_HIDDEN_MULTIPLIER = 0
CONFIG.INDIRECT_INPUT_MULTIPLIER = 0

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2 or args[0] == '-onnx_model':
        print("Usage: python3 bounds_computation.py -onnx_model <path to onnx model to compute bounds for>")
        exit(1)

    for latent_dims in [32, 64]:
        print("latent_dims:", latent_dims)
        for mtype in ["tiny", "small", "deep"]:
            model_path = args[1]

            parser = ONNXParser(model_path)
            model = parser.to_pytorch()
            jsip = JSIP(model,
                        torch.LongTensor((latent_dims, )),
                        use_ssip=True,
                        lazy_bound_comp=True)
            
            max_bounds = []
            noise = np.random.random(10)
            for eps in [1e-3, 1e-2, 5e-2, 1e-1, 5e-1]:
                maxes = []
                for n in noise:
                    bounds = torch.zeros((latent_dims, 2)).double()
                    bounds[:, 0] = (1-n)*eps
                    bounds[:, 1] = (1+n)*eps

                    res = jsip.calc_bounds(bounds.unsqueeze(0))
                    post = jsip.get_bounds_concrete_post(-1)[0]
                    bdiff = post[:, 1] - post[:, 0]
                    print("eps bounds: ", eps, torch.min(bdiff).item(), torch.mean(bdiff).item(), torch.median(bdiff).item(), torch.max(bdiff).item())
                    maxes.append(torch.max(bdiff).item())
                max_bounds.append(np.max(maxes))
            print("model:", mtype, max_bounds)
