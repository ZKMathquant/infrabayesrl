from dataclasses import dataclass
from .integration import gauss_hermite_quadrature
from typing import cast, Callable, Optional, Union
import torch as th
import torch.distributions as thd

@dataclass
class SaMeasure:
    """Scale-and-bias transformed probability measure."""
    mu: thd.Distribution
    scale: Optional[th.Tensor] = None
    bias: Optional[th.Tensor] = None

    def entropy(self) -> th.Tensor:
        """Entropy of underlying probability distribution."""
        return self.mu.entropy()

    def __add__(self, other: Union[float, "SaMeasure", th.Tensor]) -> "SaMeasure":
        if isinstance(other, (float, th.Tensor)):
            new_bias = self.bias + other if self.bias is not None else other
            return SaMeasure(self.mu, self.scale, th.as_tensor(new_bias))
        elif isinstance(other, SaMeasure):
            raise NotImplementedError("Addition of sa-measures not implemented")
        else:
            raise TypeError(f"Cannot add {type(other)} to SaMeasure")

    def __call__(self, f: Callable[[th.Tensor], th.Tensor], n: int = 20) -> th.Tensor:
        """Compute expected value of function wrt this sa-measure."""
        if isinstance(self.mu, thd.Bernoulli):
            p = cast(th.Tensor, self.mu.probs)
            E = p * f(th.ones_like(p)) + (1 - p) * f(th.zeros_like(p))
        elif isinstance(self.mu, thd.Normal):
            E = gauss_hermite_quadrature(self.mu, f, n=n)
        else:
            raise NotImplementedError(f"Expected values not implemented for {type(self.mu)}")

        if self.scale is not None:
            E = E * self.scale
        if self.bias is not None:
            E = E + self.bias
        return E

    def __mul__(self, scalar: Union[float, th.Tensor]) -> "SaMeasure":
        """Scalar multiplication of sa-measures."""
        new_scale = self.scale * scalar if self.scale is not None else scalar
        new_bias = self.bias * scalar if self.bias is not None else None
        return SaMeasure(self.mu, new_scale, new_bias)
