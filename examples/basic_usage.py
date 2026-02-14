"""Basic usage of infrabayesian components."""
import torch as th
import torch.distributions as thd
from src import InfraPolytope, SaMeasure

def demonstrate_sa_measures():
    """Show basic SaMeasure functionality."""
    print("=== SaMeasure Demo ===")
    
    # Basic usage
    mu = thd.Normal(0.0, 1.0)
    sa_measure = SaMeasure(mu)
    result = sa_measure(lambda x: x**2)
    print(f"E[X^2] for N(0,1): {result:.3f}")
    
    # With scale and bias
    scaled_measure = SaMeasure(mu, scale=th.tensor(2.0), bias=th.tensor(1.0))
    result = scaled_measure(lambda x: x)
    print(f"E[2X + 1] for N(0,1): {result:.3f}")

def demonstrate_infradistributions():
    """Show InfraPolytope functionality."""
    print("\n=== InfraPolytope Demo ===")
    
    # Create batch of distributions
    means = th.tensor([0.0, 1.0, 2.0])
    batch_normal = thd.Normal(means, th.ones(3))
    infra_dist = InfraPolytope(batch_normal)
    
    # Min expectation
    result = infra_dist(lambda x: x)
    print(f"Min E[X] over {means.tolist()}: {result:.3f}")
    
    # Quadratic function
    result = infra_dist(lambda x: x**2)
    print(f"Min E[X^2] over distributions: {result:.3f}")

def newcomb_utility_demo():
    """Demonstrate utility evaluation for Newcomb's problem."""
    print("\n=== Newcomb Utility Demo ===")
    
    # Uncertain predictor accuracy
    accuracies = th.tensor([0.8, 0.9, 0.95])
    batch_bernoulli = thd.Bernoulli(accuracies)
    infra_predictor = InfraPolytope(batch_bernoulli)
    
    def one_box_utility(prediction):
        return th.where(prediction > 0.5, 1000000.0, 0.0)
    
    def two_box_utility(prediction):
        return th.where(prediction > 0.5, 1001000.0, 1000.0)
    
    one_box_value = infra_predictor(one_box_utility)
    two_box_value = infra_predictor(two_box_utility)
    
    print(f"One-box infrabayesian value: {one_box_value:.0f}")
    print(f"Two-box infrabayesian value: {two_box_value:.0f}")
    print(f"Optimal choice: {'One-box' if one_box_value > two_box_value else 'Two-box'}")

if __name__ == "__main__":
    demonstrate_sa_measures()
    demonstrate_infradistributions()
    newcomb_utility_demo()
