import torch as th
import torch.distributions as thd
from src.integration import gauss_hermite_quadrature, monte_carlo_expectation
import pytest

@pytest.mark.parametrize("exponent", [1, 2, 3])
def test_gauss_hermite(exponent):
    """Test Gauss-Hermite quadrature accuracy."""
    th.manual_seed(0)
    
    mean = th.linspace(-1, 1, 5)
    std = th.linspace(0.1, 1, 5)
    mu = thd.Normal(mean, std)
    
    mc = monte_carlo_expectation(mu, lambda x: x ** exponent, n=100000)
    quad = gauss_hermite_quadrature(mu, lambda x: x ** exponent)
    
    assert th.allclose(mc, quad.float(), atol=0.05)

def test_monte_carlo_convergence():
    """Test Monte Carlo convergence."""
    mu = thd.Normal(0.0, 1.0)
    
    # Should converge to 1 for E[X^2] with N(0,1)
    result = monte_carlo_expectation(mu, lambda x: x**2, n=10000)
    assert abs(result.item() - 1.0) < 0.1
