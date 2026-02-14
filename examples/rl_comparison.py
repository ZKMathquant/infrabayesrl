import torch as th
import matplotlib.pyplot as plt
from src.ib_rl_agent import InfrabayesianRLAgent, ClassicalRLAgent
from src.newcomb_env import NewcombEnvironment, LogicalPredictorEnv

def run_experiment(agent, env, episodes: int = 1000):
    """Run RL experiment and collect results."""
    rewards = []
    actions = []
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        agent.update(state, action, reward, next_state)
        
        rewards.append(reward)
        actions.append(action)
        
        if episode % 200 == 0:
            recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
            recent_actions = actions[-100:] if len(actions) >= 100 else actions
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            one_box_rate = sum(1 for a in recent_actions if a == 0) / len(recent_actions)
            print(f"Episode {episode}: Reward={avg_reward:.0f}, One-box={one_box_rate:.2f}")
    
    return rewards, actions

def compare_agents():
    """Compare classical vs infrabayesian RL on Newcomb's problem."""
    print("=== Newcomb's Problem: Classical vs Infrabayesian RL ===\n")
    
    # Test environments
    envs = {
        'Standard': NewcombEnvironment(0.9),
        'Logical': LogicalPredictorEnv(0.95)
    }
    
    for env_name, env in envs.items():
        print(f"\n--- {env_name} Predictor ---")
        
        # Classical agent
        classical = ClassicalRLAgent([0, 1], learning_rate=0.1, epsilon=0.1)
        print("Classical RL:")
        classical_rewards, classical_actions = run_experiment(classical, env, 1000)
        
        # Reset environment
        env.agent_history = []
        env.episode_count = 0
        
        # Infrabayesian agent
        ib_agent = InfrabayesianRLAgent([0, 1], uncertainty_radius=0.2, 
                                       learning_rate=0.1, epsilon=0.1)
        print("\nInfrabayesian RL:")
        ib_rewards, ib_actions = run_experiment(ib_agent, env, 1000)
        
        # Final analysis
        print(f"\n{env_name} Results (last 200 episodes):")
        
        classical_final = classical_actions[-200:]
        classical_one_box = sum(1 for a in classical_final if a == 0) / len(classical_final)
        classical_avg_reward = sum(classical_rewards[-200:]) / 200
        
        ib_final = ib_actions[-200:]
        ib_one_box = sum(1 for a in ib_final if a == 0) / len(ib_final)
        ib_avg_reward = sum(ib_rewards[-200:]) / 200
        
        print(f"Classical: One-box={classical_one_box:.3f}, Reward={classical_avg_reward:.0f}")
        print(f"Infrabayesian: One-box={ib_one_box:.3f}, Reward={ib_avg_reward:.0f}")
        print(f"Optimal: One-box=1.000, Reward=1000000")

if __name__ == "__main__":
    compare_agents()
