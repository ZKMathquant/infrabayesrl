"""Compare classical vs infrabayesian RL on Newcomb's problem."""
import torch as th
from src import InfrabayesianRLAgent, ClassicalRLAgent, NewcombEnvironment, LogicalPredictorEnv
from src.utils import run_experiment, plot_comparison_results

def compare_on_newcomb():
    """Main comparison function."""
    print("=== Newcomb's Problem: Classical vs Infrabayesian RL ===\n")
    
    # Test different environments
    environments = {
        'Standard Predictor (90% accuracy)': NewcombEnvironment(0.9),
        'Logical Predictor (95% accuracy)': LogicalPredictorEnv(0.95)
    }
    
    results = {}
    
    for env_name, env in environments.items():
        print(f"\n--- {env_name} ---")
        
        # Classical RL agent
        print("Training Classical RL Agent...")
        classical = ClassicalRLAgent([0, 1], learning_rate=0.1, epsilon=0.1)
        classical_results = run_experiment(classical, env, episodes=1000, verbose=False)
        
        # Reset environment
        env.agent_history = []
        env.episode_count = 0
        
        # Infrabayesian RL agent
        print("Training Infrabayesian RL Agent...")
        ib_agent = InfrabayesianRLAgent([0, 1], uncertainty_radius=0.2, 
                                       learning_rate=0.1, epsilon=0.1)
        ib_results = run_experiment(ib_agent, env, episodes=1000, verbose=False)
        
        # Store results
        results[env_name] = {
            'classical': classical_results,
            'infrabayesian': ib_results
        }
        
        # Print summary
        print_summary(env_name, classical_results, ib_results)
        
        # Plot comparison
        plot_comparison_results(classical_results, ib_results, 
                              save_path=f"experiments/results/{env_name.replace(' ', '_')}_comparison.png")

def print_summary(env_name: str, classical_results: dict, ib_results: dict):
    """Print summary of results."""
    print(f"\n{env_name} Results (final 200 episodes):")
    
    # Classical results
    classical_final = classical_results['actions'][-200:]
    classical_rewards = classical_results['rewards'][-200:]
    classical_one_box = sum(1 for a in classical_final if a == 0) / len(classical_final)
    classical_avg_reward = sum(classical_rewards) / len(classical_rewards)
    
    # Infrabayesian results
    ib_final = ib_results['actions'][-200:]
    ib_rewards = ib_results['rewards'][-200:]
    ib_one_box = sum(1 for a in ib_final if a == 0) / len(ib_final)
    ib_avg_reward = sum(ib_rewards) / len(ib_rewards)
    
    print(f"Classical RL:")
    print(f"  One-boxing rate: {classical_one_box:.3f}")
    print(f"  Average reward: {classical_avg_reward:.0f}")
    print(f"  Optimality gap: {abs(1000000 - classical_avg_reward)/1000000:.3f}")
    
    print(f"Infrabayesian RL:")
    print(f"  One-boxing rate: {ib_one_box:.3f}")
    print(f"  Average reward: {ib_avg_reward:.0f}")
    print(f"  Optimality gap: {abs(1000000 - ib_avg_reward)/1000000:.3f}")
    
    print(f"Theoretical Optimal:")
    print(f"  One-boxing rate: 1.000")
    print(f"  Average reward: 1000000")
    print(f"  Optimality gap: 0.000")

if __name__ == "__main__":
    import os
    os.makedirs("experiments/results", exist_ok=True)
    compare_on_newcomb()
