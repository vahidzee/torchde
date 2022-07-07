import torch
import typing as th
import functools


class OrderedLayerMixin1D:
    """
    Mixin to support ordered 1D outputs that maintain an autoregressive data flow.

    This class should be initialized after `torch.nn.Module` is inherited and
    initialized in the desired module.

    Attributes:
        ordering: Current ordering of the output neurons.
        auto_connection: Whether to allow equal label connections.
        mask:
            A zero-one 2D mask [OUTPUT x INPUT] matrix defining the connectivity
            of output and input neurons.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        auto_connection: bool = True,
        device: th.Optional[torch.device] = None,
        mask_dtype: torch.dtype = torch.uint8,
    ) -> None:
        """
        Initiates ordering related buffers.

        Args:
            in_featuers: Number of input dimensions with ordering.
            out_features: Number of output dimensions with ordering.
            auto_connection:
                A boolean value specifying whether output neurons with the
                same labels as the input neurons can be connected.
            device: The device to instanciate the buffers on.
            masked_dtype: Data type for mask matrix.
        Retures:
            None
        """
        self.auto_connection = auto_connection
        self.register_buffer("ordering", torch.empty(out_features, device=device, dtype=torch.int))
        self.register_buffer(
            "mask",
            torch.ones(out_features, in_features, device=device, dtype=mask_dtype),
        )

    @functools.cached_property
    def connection_operator(self):
        return torch.less_equal if self.auto_connection else torch.less

    def reorder(
        self,
        inputs_ordering: torch.IntTensor,
        ordering: th.Optional[torch.IntTensor] = None,
        allow_detached_neurons: bool = True,
        highest_ordering_label: th.Optional[int] = None,
        generator: th.Optional[torch.Generator] = None,
    ) -> None:
        """
        (Re)computes the output ordering and the mask matrix.

        This function computes the layer's ordering based on the given
        ordering (to be enforced) or the ordering of layer's inputs and a
        random number generator. Optionally you can choose to disallow
        detached neurons, so that the layer ordering labels are chosen
        from higher values than the last layer.

        Args:
            inputs_ordering:
                Ordering of inputs to the layer used for computing new
                layer inputs (randomly out with the generator), and the
                connectivity mask.
            ordering: An optional ordering to be enforced
            allow_detached_neurons:
                If true, the minimum label for this layer's outputs would
                start from zero regardless of whether the previous layer's
                ordering. Else, layer's labels will start from the minimum
                label of inputs.
            highest_ordering_label: Maximum label to use.
            generator: Random number generator.

        Returns:
            None
        """
        if ordering is not None:
            # enforcing ordering and if needed repeating the providded ordering
            # across layer's ordering. (especially used for predicting autoregressive
            # mixture of parameters which requires an output with a size with a multiple
            # of the number of input dimensions.)
            self.ordering.data.copy_(torch.repeat_interleave(ordering, self.ordering.shape[0] // ordering.shape[0]))
        else:
            self.ordering.data.copy_(
                torch.randint(
                    low=0 if allow_detached_neurons else inputs_ordering.min(),
                    high=highest_ordering_label or inputs_ordering.max(),
                    size=self.ordering.shape,
                    generator=generator,
                )
            )
        self.mask.data.copy_(self.connection_operator(inputs_ordering[:, None], self.ordering[None, :]).T)
