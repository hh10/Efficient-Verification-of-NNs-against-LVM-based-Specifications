# Patches to apply to open-source VeriNet to run parent codebase
1. Clone verinet folder from https://github.com/vas-group-imperial/VeriNet repo.
2. Add verification support for ConvTranspose2d by copying (a+) [linear.py](linear.py) to verinet/sip_torch/operations/linear.py
3. Add verification support for LeakyReLU by copying (a+) [piecewise_linear.py](piecewise_linear.py) to verinet/sip_torch/operations/piecewise_linear.py
4. Apply verifier.py patch to ensure propagation in doubles instead of floats, since necessary for invertibility.
