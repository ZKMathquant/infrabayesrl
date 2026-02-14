import torch as th
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any

def run_experiment(agent, env, episodes: int = 1000, verbose: bool = True) -> Dict[str, List]:
    """Run RL experiment and collect comprehensive results."""
    rewards = []
    actions = []
    predictions = []
    q_values_history = []
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        agent.update(state, action, reward, next_state)
        
        rewards.append(reward)
        actions.append(action)
        predictions.append(info.get('predicted', -1))
        
        # Track Q-values for analysis
        if hasattr(agent, 'q_values'):
            q_vals = {a: agent.q_values[state][a] for a in agent.actions}
            q_values_history.append(q_vals.copy())
        
        if verbose and episode % 200 == 0 and episode > 0:
            recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
            recent_actions = actions[-100:] if len(actions) >= 100 else actions
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            one_box_rate = sum(1 for a in recent_actions if a == 0) / len(recent_actions)
            print(f"Episode {episode}: Reward={avg_reward:.0f}, One-box={one_box_rate:.2f}")
    
    return {
        'rewards': rewards,
        'actions': actions,
        'predictions': predictions,
        'q_values': q_values_history,
        'final_policy': get_final_policy(agent),
        'convergence_metrics': calculate_convergence_metrics(actions, rewards)
    }

def get_final_policy(agent) -> Dict[int, float]:
    """Extract final policy from agent."""
    if hasattr(agent, 'q_values'):
        state = 0  # Single state in Newcomb
        policy = {}
        for action in agent.actions:
            policy[action] = agent.q_values[state][action]
        return policy
    return {}

def calculate_convergence_metrics(actions: List[int], rewards: List[float]) -> Dict[str, float]:
    """Calculate convergence and performance metrics."""
    if len(actions) < 100:
        return {}
    
    final_100 = actions[-100:]
    one_box_rate = sum(1 for a in final_100 if a == 0) / len(final_100)
    
    final_rewards = rewards[-100:]
    avg_reward = sum(final_rewards) / len(final_rewards)
    
    # Measure consistency (lower variance = more consistent)
    consistency = 1.0 - np.var(final_100)
    
    return {
        'final_one_box_rate': one_box_rate,
        'final_avg_reward': avg_reward,
        'consistency': consistency,
        'optimal_gap': abs(1000000.0 - avg_reward) / 1000000.0
    }

def plot_comparison_results(classical_results: Dict, ib_results: Dict, save_path: str = None):
    """Plot comparison between classical and infrabayesian agents."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards over time
    window = 50
    classical_rewards = moving_average(classical_results['rewards'], window)
    ib_rewards = moving_average(ib_results['rewards'], window)
    
    ax1.plot(classical_rewards, label='Classical RL', alpha=0.8)
    ax1.plot(ib_rewards, label='Infrabayesian RL', alpha=0.8)
    ax1.set_title('Average Reward Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Action rates over time
    classical_one_box = moving_average([1 if a == 0 else 0 for a in classical_results['actions']], window)
    ib_one_box = moving_average([1 if a == 0 else 0 for a in ib_results['actions']], window)
    
    ax2.plot(classical_one_box, label='Classical RL', alpha=0.8)
    ax2.plot(ib_one_box, label='Infrabayesian RL', alpha=0.8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Optimal')
    ax2.set_title('One-Boxing Rate Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('One-Boxing Rate')
    ax2.legend()
    ax2.grid(True)
    
    # Final policy comparison
    episodes = len(classical_results['actions'])
    final_window = min(200, episodes // 4)
    
    classical_final = classical_results['actions'][-final_window:]
    ib_final = ib_results['actions'][-final_window:]
    
    classical_one_box_final = sum(1 for a in classical_final if a == 0) / len(classical_final)
    ib_one_box_final = sum(1 for a in ib_final if a == 0) / len(ib_final)
    
    ax3.bar(['Classical RL', 'Infrabayesian RL', 'Optimal'], 
            [classical_one_box_final, ib_one_box_final, 1.0],
            color=['blue', 'orange', 'red'], alpha=0.7)
    ax3.set_title('Final One-Boxing Rates')
    ax3.set_ylabel('One-Boxing Rate')
    ax3.set_ylim(0, 1.1)
    
    # Reward distribution
    classical_final_rewards = classical_results['rewards'][-final_window:]
    ib_final_rewards = ib_results['rewards'][-final_window:]
    
    ax4.hist(classical_final_rewards, bins=20, alpha=0.6, label='Classical RL', density=True)
    ax4.hist(ib_final_rewards, bins=20, alpha=0.6, label='Infrabayesian RL', density=True)
    ax4.axvline(x=1000000, color='red', linestyle='--', alpha=0.8, label='Optimal Reward')
    ax4.set_title('Final Reward Distribution')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Density')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def moving_average(data: List[float], window: int) -> List[float]:
    """Calculate moving average."""
    if len(data) < window:
        return data
    
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    
    return result

def save_results(results: Dict[str, Any], filename: str):
    """Save experimental results."""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filename: str) -> Dict[str, Any]:
    """Load experimental results."""
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)
