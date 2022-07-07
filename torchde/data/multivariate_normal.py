import torch
import typing as th


class MultivariateGaussianDataset(torch.utils.data.Dataset):
    """Toy dataset of Multivariate Gaussian Distribution samples

    The distribution is either a single Gaussian or a mixture of Gaussians.

    Attributes:
        distribution: distribution objects
        data: samples (used for training/validation)
        length: length of the dataset
    """

    def __init__(
        self,
        loc: torch.Tensor,
        covariance_matrix: torch.Tensor,
        mixture_logits: th.Optional[torch.Tensor] = None,
        length: int = 5000,
        **kwargs, # ignore kwargs
    ) -> None:
        """
        Args:
            loc: mean of the distribution of shape ([NUM_Mixtures], DIMS)
            covariance_matrix: covariance matrix of the distribution of shape ([NUM_Mixtures], DIMS, DIMS)
            mixture_logits: logits of the mixture distribution of shape ([NUM_Mixtures], )
            length: length of the dataset
        """
        super().__init__()
        if len(loc.shape) == 1:
            self.distribution = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
        else:
            component_distributions = torch.distributions.MultivariateNormal(
                loc=loc, covariance_matrix=covariance_matrix
            )
            mixture_distributions = torch.distributions.Categorical(
                logits=mixture_logits if mixture_logits is not None else torch.ones(loc.shape[0])
            )
            self.distribution = torch.distributions.MixtureSameFamily(mixture_distributions, component_distributions)
        self.data = self.distribution.sample((length,))
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]
