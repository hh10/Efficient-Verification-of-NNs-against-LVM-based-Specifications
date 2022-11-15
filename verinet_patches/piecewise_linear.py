# Code patch to add to https://github.com/vas-group-imperial/VeriNet/blob/main/verinet/sip_torch/operations/piecewise_linear.py

class LeakyRelu(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["negative_slope"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.modules.activation.LeakyReLU, nn.LeakyReLU]

    @property
    def has_cex_optimisable_relaxations(self) -> bool:

        """
        Returns true if the op-type can take advantage of cex-optimised relaxations.
        """

        return True

    def get_num_non_linear_neurons(self, bounds_concrete_pre: torch.Tensor) -> int:

        """
        Returns the number of non-linear neurons based on bounds.

        Note that if bounds_concrete_pre.shape[0] != weights.shape[0], it is assumed
        that all neurons are non-linear wrt prelu parameters (prelu.weight != 1 for
        any of the given bounds).

        Args:
            bounds_concrete_pre:
                A Nx2 tensor with the lower bounds of the neurons in the first col
                and upper bounds in the second.
        """

        weights = self.params["negative_slope"].flatten()
        in_shape = self.params["in_shape"]

        if len(in_shape) > 1:
            weights = torch.repeat_interleave(weights, torch.prod(in_shape[1:]).to(device=weights.device))

        if weights.shape[0] == bounds_concrete_pre.shape[0]:
            non_lin_neurons = torch.abs(weights - 1) > 1e-5
            non_lin_weights = (bounds_concrete_pre[:, 0] < 0) * (bounds_concrete_pre[:, 1] > 0)

            return int(torch.sum(non_lin_neurons * non_lin_weights))

        else:
            return int(torch.sum((bounds_concrete_pre[:, 0] < 0) * (bounds_concrete_pre[:, 1] > 0)))

    def forward(self, x: torch.Tensor, add_bias: bool = True):

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        weights = self.params["negative_slope"]
        x[x < 0] = x[x < 0] * weights[x < 0]

        return x

    def ibp_batch_forward(self, bounds_pre: List[torch.Tensor], add_bias: bool = True):

        """
        Propagates batches of concrete interval bounds through the network.

        Args:
            bounds_pre:
                The constraints on the input of shape BxNx2. B is the batch dim,
                N the nodes and lower and upper bounds in the last dim.
            add_bias:
                If true, the bias is added.
        Returns:
            The output value.
        """
        if len(bounds_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_pre)}")

        bounds_post = bounds_pre[0].clone()
        bounds_post[bounds_post < 0] *= self.params["negative_slope"]
        return bounds_post

    def ssip_batch_forward(self, bounds_symbolic_pre: List[torch.Tensor]) -> torch.Tensor:

        """
        Propagates the upper and lower bounding equations of ssip.

        Args:
            bounds_symbolic_pre:
                The symbolic bounds. A tensor of dim 2xBxNxM where B is the
                batch dim, N is the number of equations (or nodes in the
                layer), M is the number of coefficients (or input dim) and
                the first dim contains the lower and upper bounds, respectively.
        Returns:
            The post-op values
        """

        raise RuntimeError(f"ssip_batch_forward not implemented for non-linear function")

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        raise NotImplementedError(f"propagate_reversed(...) not implemented")

    def rsip_backward_batch(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates batches of symbolic bounds by substituting variables
        from the  previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A BxExM tensor where the first dim contains the batches, second
                dim the equations and third dim the coefficients.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        raise NotImplementedError()

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Calculates the linear relaxation

        The linear relaxation is a 2xNx2 tensor, where relaxation[0] represents a and b
        of the lower relaxation equation: l(const_terms) = ax + b (correspondingly,
        relaxation[1] is the upper relaxation).

        The relaxations are described in detail in the DeepSplit paper.

        Args:
            lower_bounds_concrete_in:
                The concrete lower bounds of the input to the nodes
            upper_bounds_concrete_in:
                The concrete upper bounds of the input to the nodes
            force_parallel:
                If true, parallel relaxations are required.
        Returns:
            The relaxations as a Nx2 tensor
        """

        weights = self.params["negative_slope"]
        in_shape = self.params["in_shape"]
        if len(in_shape) > 1:
            weights = torch.repeat_interleave(weights, torch.prod(in_shape[1:]).to(device=weights.device))

        layer_size = lower_bounds_concrete_in.shape[0]
        relaxations = new_tensor((2, layer_size, 2), device=lower_bounds_concrete_in.device, dtype=self._precision)

        # Operating in the positive area
        fixed_upper_idx = torch.nonzero(lower_bounds_concrete_in >= 0)
        relaxations[:, fixed_upper_idx, 0] = 1
        relaxations[:, fixed_upper_idx, 1] = 0

        # Operating in the negative area
        fixed_lower_idx = torch.nonzero(upper_bounds_concrete_in <= 0)
        relaxations[:, fixed_lower_idx, 0] = weights
        relaxations[:, fixed_lower_idx, 1] = 0

        # Operating in the non-linear area
        mixed_idx = torch.nonzero((upper_bounds_concrete_in > 0)*(lower_bounds_concrete_in < 0))

        if len(mixed_idx) == 0:
            return relaxations

        xl = lower_bounds_concrete_in[mixed_idx]
        xu = upper_bounds_concrete_in[mixed_idx]

        # Upper relaxation
        a = (xu - xl*weights) / (xu - xl)
        b = xl*weights - a * xl
        relaxations[1, :, 0][mixed_idx] = a
        relaxations[1, :, 1][mixed_idx] = b

        # Lower relaxation
        if force_parallel:
            relaxations[0, :, 0][mixed_idx] = a
        else:
            larger_up_idx = (upper_bounds_concrete_in + lower_bounds_concrete_in) > 0
            smaller_up_idx = ~larger_up_idx

            relaxations[0, larger_up_idx, 0] = 1
            relaxations[0, smaller_up_idx, 0] = weights

        return relaxations

    def linear_relaxation_batch(self, concrete_bounds: list, force_parallel: bool = False) -> torch.Tensor:

        """
        Calculates the linear relaxations for a batch of inputs.

        The linear relaxation is a 2xBxNx2 tensor, where dim 0 contains the
        lower/upper relaxations, dim 1 the batches, dim 2 the nodes, and dim 3
        a and b in the relaxation ax + b.

        Args:
            concrete_bounds:
                A BxNx2 tensor with the concrete bounds.
            force_parallel:
                If true, parallel relaxations are used.
        Returns:
            The relaxations as a 2xBxNx2 tensor
        """
        if len(concrete_bounds) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(concrete_bounds)}")
        concrete_bounds = concrete_bounds[0]

        org_shape = concrete_bounds.shape
        concrete_bounds = concrete_bounds.reshape(-1, 2)

        relaxations = new_tensor((concrete_bounds.shape[0], 2, 2), device=concrete_bounds.device,
                                 dtype=self._precision, fill=0)

        # Operating in the positive area
        relaxations[concrete_bounds[:, 0] > 0, :, 0] = 1
        relaxations[concrete_bounds[:, 0] < 0, :, 0] = self.params["negative_slope"]

        # Operating in the non-linear area
        mixed_idx = ((concrete_bounds[:, 1] >= 0) * (concrete_bounds[:, 0] <= 0) *
                     ((concrete_bounds[:, 1] - concrete_bounds[:, 0]) > 0))

        if torch.sum(mixed_idx) == 0:
            return relaxations.reshape((*org_shape[0:2], 2, 2))

        xl, xu = concrete_bounds[mixed_idx, 0], concrete_bounds[mixed_idx, 1]

        # Upper relaxation
        a = (xu - self.params["negative_slope"] * xl) / (xu - xl)
        relaxations[mixed_idx, 0, 0] = a  # relaxations are to be necessarily parallelograms?
        relaxations[mixed_idx, 1, 0] = a  # relaxations are to be necessarily parallelograms?
        b = (1 - a) * xu
        relaxations[mixed_idx, 1, 1] = b if self.params["negative_slope"] < 1 else 0
        relaxations[mixed_idx, 0, 1] = 0 if self.params["negative_slope"] < 1 else b

        return relaxations.reshape((*org_shape[0:2], 2, 2))

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def optimise_linear_relaxation(self, relaxation: torch.Tensor,
                                   bounds: torch.Tensor,
                                   values: torch.Tensor) -> torch.Tensor:

        """
        Optimised the given linear relaxation with respect to a set of values as
        calculated by the neural network.

        Args:
            relaxation:
                The current linear relaxation.
            bounds:
                The concrete pre-activation bounds for the node.
            values:
                The values of the node as calculated by the neural network
        Returns:
            The relaxations as a Nx2 array
        """

        weights = self.params["negative_slope"]
        in_shape = self.params["in_shape"]
        if len(in_shape) > 1:
            weights = torch.repeat_interleave(weights, torch.prod(in_shape[1:]).to(device=weights.device))

        pos_idx = values > 0
        neg_idx = values <= 0

        relaxation[0, pos_idx, 0] = 1
        relaxation[0, neg_idx, 0] = weights
        relaxation[0, :, 1] = 0

        max_bounds_multiple = CONFIG.OPTIMISED_RELU_RELAXATION_MAX_BOUNDS_MULTIPLIER

        bounds_ratio = bounds[0][:, 1] / abs(bounds[0][:, 0])
        relaxation[0, bounds_ratio < 1/max_bounds_multiple, 0] = weights
        relaxation[0, bounds_ratio > max_bounds_multiple, 0] = 1

        return relaxation

    def split_point(self, xl: float, xu: float):

        """
        Returns the preferred split point for branching which is 0 for the ReLU.

        Args:
            xl:
                The lower bound on the input.
            xu:
                The upper bound on the input.

        Returns:
            The preferred split point
        """

        return torch.zeros(1)

    def get_non_linear_neurons(self, bounds_concrete_pre: torch.Tensor) -> torch.Tensor:

        """
        An array of boolean values. 'True' indicates that the corresponding neuron
        is non-linear in the input domain as described by bounds_concrete_pre; 'false'
        that it is linear.

        Note that if bounds_concrete_pre.shape[0] != weights.shape[0], it is assumed
        that all neurons are non-linear wrt prelu parameters (prelu.weight != 1 for
        any of the given bounds).

        Args:
            bounds_concrete_pre:
                The concrete pre-operation value.
        Returns:
            A boolean tensor with 'true' for non-linear neurons and false otherwise.
        """

        weights = self.params["negative_slope"]
        in_shape = self.params["in_shape"]

        if len(in_shape) > 1:
            weights = torch.repeat_interleave(weights, torch.prod(in_shape[1:]).to(device=weights.device))

        return (bounds_concrete_pre[:, 0] < 0) * (bounds_concrete_pre[:, 1] > 0)

    def forward_prop_ssip_relaxation_batch(self,
                                           symb_bounds_pre: torch.Tensor,
                                           bounds_concrete_pre: torch.Tensor,
                                           relaxations: torch.Tensor) -> torch.Tensor:

        """
        Forward-propagates batched symbolic equations through linear relaxations.

        Args:
            symb_bounds_pre:
                A 2xBxEx(N+1) tensor with the symbolic bounds. Here, B is the
                batch dim, E the number of equations (number of nodes in
                current layer) and N is the number of coefficients
                (number of input nodes). +1 in the last dim is due to the
                constant term.
            bounds_concrete_pre:
                A BxNx2 tensor. Here B, N are as in bounds_symbolic_post,
                the last dim contains the lower and upper bounds.
            relaxations:
                A 2xBxEx2x2 Here B, E are as in bounds_symbolic_post, the fourth
                dim contains the lower and upper relaxation while the last
                dim contains the coefficients a, b in the relaxation ax + b.
                The first dim contains the relaxation wrt the lower and upper
                symbolic bounds, respectively.

        Returns:
            The resulting symbolic bounds of shape 2xBxEx(N+1).
        """
        if len(symb_bounds_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(symb_bounds_pre)}")

        symb_bounds_pre = symb_bounds_pre[0]

        output_bounds = new_tensor(symb_bounds_pre.shape, device=symb_bounds_pre.device, dtype=self._precision,
                                   fill=0)

        output_bounds[0] = symb_bounds_pre[0] * relaxations[0, ..., 0, 0:1]
        output_bounds[0, ..., -1] += relaxations[0, ..., 0, 1]

        output_bounds[1] = symb_bounds_pre[1] * relaxations[1, ..., 1, 0:1]
        output_bounds[1, ..., -1] += relaxations[1, ..., 1, 1]

        return output_bounds

    def backprop_through_relaxation_batch(self,
                                          symb_bounds_post: torch.Tensor,
                                          bounds_concrete_pre: torch.Tensor,
                                          relaxations: torch.Tensor,
                                          lower: bool = True) -> torch.Tensor:

        """
        Back-propagates batched inputs through the node using relaxations.

        Args:
            symb_bounds_post:
                A BxEx(N+1) tensor with the symbolic bounds. Here, B is the batch
                dim, E the number of bounds and C the number of nodes in the
                current layer (N+1 is coefficients + one constant term).
            bounds_concrete_pre:
                A BxNx2 tensor. Here B, N are as in bounds_symbolic_post,
                the last dim contains the lower and upper bounds.
            relaxations:
                A BxNx2x2 Here B, N are as in bounds_symbolic_post, the third
                dim contains the lower and upper relaxation while the last
                dim contains the coefficients a, b in the relaxation ax + b.
            lower:
                If true, the lower bounds are calculated, else the upper.

        Returns:
            The resulting symbolic bounds.
        """

        raise RuntimeError("Tried back propagating through relaxation for a linear function")

    def out_shape(self, in_shapes: List[torch.LongTensor]) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        if not len(in_shapes) == 1:
            raise ValueError("Expected one set of inputs")

        return in_shapes[0]