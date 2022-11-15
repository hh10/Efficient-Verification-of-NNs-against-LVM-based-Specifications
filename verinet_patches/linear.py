# Code patch to add to https://github.com/vas-group-imperial/VeriNet/blob/main/verinet/sip_torch/operations/linear.py

class ConvTranspose2d(AbstractOperation):

    def __init__(self):

        self._out_unfolded = None
        super().__init__()

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return ["weight", "bias", "kernel_size", "padding", "output_padding",
                "stride", "in_channels", "out_channels", "groups"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.ConvTranspose2d]

    def ibp_forward(self, bounds_pre: list, add_bias: bool = True):

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
        bounds_pre = bounds_pre[0]

        stride = self.params["stride"]
        padding = self.params["padding"]
        output_padding = self.params["output_padding"]
        weights = self.params["weight"]
        bias = self.params["bias"]
        groups = self.params["groups"]
        in_shape = self.params["in_shape"][0]

        pos_weights, neg_weights = weights.clone(), weights.clone()
        pos_weights[pos_weights < 0] = 0
        neg_weights[neg_weights > 0] = 0

        low_pre = bounds_pre[..., 0].reshape((bounds_pre.shape[0], *in_shape))
        up_pre = bounds_pre[..., 1].reshape((bounds_pre.shape[0], *in_shape))

        low_post = functional.conv_transpose2d(low_pre, weight=pos_weights, stride=stride, padding=padding,
                                               output_padding=output_padding, groups=groups)
        low_post += functional.conv_transpose2d(up_pre, weight=neg_weights, stride=stride, padding=padding,
                                                output_padding=output_padding, groups=groups)
        up_post = functional.conv_transpose2d(low_pre, weight=neg_weights, stride=stride, padding=padding,
                                              output_padding=output_padding, groups=groups)
        up_post += functional.conv_transpose2d(up_pre, weight=pos_weights, stride=stride, padding=padding,
                                               output_padding=output_padding, groups=groups)

        if add_bias and bias is not None:
            low_post += bias.view(1, -1, 1, 1)
            up_post += bias.view(1, -1, 1, 1)

        low_post = low_post.reshape(bounds_pre.shape[0], -1).unsqueeze(-1)
        up_post = up_post.reshape(bounds_pre.shape[0], -1).unsqueeze(-1)

        return torch.cat((low_post, up_post), dim=-1)

    def ssip_forward(self, bounds_symbolic_pre: list) -> torch.Tensor:

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

        if len(bounds_symbolic_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_symbolic_pre)}")

        bounds_symbolic_pre = bounds_symbolic_pre[0]

        stride = self.params["stride"]
        padding = self.params["padding"]
        output_padding = self.params["output_padding"]
        weights = self.params["weight"]
        bias = self.params["bias"]
        groups = self.params["groups"]
        in_shape = self.params["in_shape"][0]

        batch_dim, eq_dim, coeff_dim = bounds_symbolic_pre.shape[1:]

        low_pre = bounds_symbolic_pre[0].permute(0, 2, 1).reshape((-1, *in_shape))
        up_pre = bounds_symbolic_pre[1].permute(0, 2, 1).reshape((-1, *in_shape))

        # Positive coefficients
        tmp_weights = weights.clone()
        tmp_weights[tmp_weights < 0] = 0

        low_post = functional.conv_transpose2d(low_pre, weight=tmp_weights, stride=stride,
                                               output_padding=output_padding, padding=padding, groups=groups)
        up_post = functional.conv_transpose2d(up_pre, weight=tmp_weights, stride=stride,
                                              output_padding=output_padding, padding=padding, groups=groups)

        # Negative coefficients
        tmp_weights[:] = weights
        tmp_weights[tmp_weights > 0] = 0

        low_post += functional.conv_transpose2d(up_pre, weight=tmp_weights, stride=stride,
                                                output_padding=output_padding, padding=padding, groups=groups)
        up_post += functional.conv_transpose2d(low_pre, weight=tmp_weights, stride=stride,
                                               output_padding=output_padding, padding=padding, groups=groups)

        low_post = low_post.view((batch_dim, coeff_dim, *low_post.shape[1:]))
        up_post = up_post.view((batch_dim, coeff_dim, *up_post.shape[1:]))

        if bias is not None:
            low_post[:, -1] += bias.view(1, -1, 1, 1)
            up_post[:, -1] += bias.view(1, -1, 1, 1)

        low_post = low_post.reshape(batch_dim, coeff_dim, -1).permute(0, 2, 1).unsqueeze(0)
        up_post = up_post.reshape(batch_dim, coeff_dim, -1).permute(0, 2, 1).unsqueeze(0)

        return torch.cat((low_post, up_post), dim=0)

    # noinspection PyTypeChecker,PyCallingNonCallable
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

        in_shape = self.params["in_shape"][0]
        weights = self.params["weight"]

        bias = self.params["bias"]
        groups = self.params["groups"]
        stride = self.params["stride"]
        padding = self.params["padding"]
        out_shape = self.out_shape([in_shape])

        num_eqs = bounds_symbolic_post.shape[0]

        bounds_symbolic_pre = new_tensor((num_eqs, torch.prod(in_shape) + 1), device=str(bounds_symbolic_post.device),
                                         dtype=self._precision, fill=None)

        bounds_symbolic_pre[:, -1] = bounds_symbolic_post[:, -1]

        if not add_bias or bias is None:
            bias = new_tensor((out_shape[0]), dtype=self._precision, device=str(bounds_symbolic_pre.device), fill=0)

        vars_pr_channel = torch.prod(self.out_shape(in_shapes=[torch.LongTensor(tuple(in_shape))])[1:3])

        bounds_symbolic_pre[:, :-1] = functional.conv2d(
            bounds_symbolic_post[:, :-1].view(bounds_symbolic_post.shape[0], *tuple(out_shape)), weight=weights,
            stride=stride, padding=padding, groups=groups).view(num_eqs, -1)

        bounds_symbolic_pre[:, -1] += torch.sum(bounds_symbolic_post[:, :-1].view(num_eqs, -1, vars_pr_channel) *
                                                bias.view(1, -1, 1), dim=(1, 2))

        return [bounds_symbolic_pre]

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

    # noinspection PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        params = self.params
        channels = params["out_channels"]

        height = ((in_shape[1] - 1) * params["stride"][0] + params["kernel_size"][0] - 2 * params["padding"][0] +
                  params["output_padding"][0])
        width = ((in_shape[2] - 1) * params["stride"][1] + params["kernel_size"][1] - 2 * params["padding"][1] +
                 params["output_padding"][1])

        return torch.LongTensor((channels, height, width))
