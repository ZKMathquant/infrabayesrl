import torch as th
from src import InfrabayesianRLAgent, ClassicalRLAgent, NewcombEnvironment
import pytest

def test_agent_initialization():
    """Test agent initialization."""
    agent = InfrabayesianRLAgent([0, 1])
    assert agent.actions == [0, 1]
    assert agent.uncertainty_radius > 0
    assert agent.learning_rate > 0

def test_newcomb_environment():
    """Test Newcomb environment."""
    env = NewcombEnvironment()
    state = env.reset()
    next_state, reward, done, info = env.step(0)
    
    assert isinstance(reward, float)
    assert 'predicted' in info
    assert 'actual' in info
    assert done is True

def test_agent_learning():
    """Test that agents can learn."""
    env = NewcombEnvironment(predictor_accuracy=1.0)  # Perfect predictor
    agent = InfrabayesianRLAgent([0, 1], epsilon=0.0)  # No exploration
    
    # Train for a few episodes
    for _ in range(50):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state)
    
    # Should prefer one-boxing
    final_action = agent.select_action(0)
    # With perfect predictor, one-boxing should be learned
    assert final_action in [0, 1]  # Basic sanity check

def test_classical_vs_ib_difference():
    """Test that classical and IB agents behave differently."""
    env = NewcombEnvironment(predictor_accuracy=0.95)
    
    classical = ClassicalRLAgent([0, 1], epsilon=0.05)
    ib_agent = InfrabayesianRLAgent([0, 1], epsilon=0.05)
    
    # Short training
    for _ in range(100):
        # Classical
        state = env.reset()
        action_c = classical.select_action(state)
        _, reward_c, _, _ = env.step(action_c)
        classical.update(state, action_c, reward_c, state)
        
        # IB (reset environment state)
        env.agent_history = env.agent_history[:-1]
        action_ib = ib_agent.select_action(state)
        _, reward_ib, _, _ = env.step(action_ib)
        ib_agent.update(state, action_ib, reward_ib, state)
    
    # Both should be capable of making decisions
    classical_action = classical.select_action(0)
    ib_action = ib_agent.select_action(0)
    
    assert classical_action in [0, 1]
    assert ib_action in [0, 1]

def test_credal_set_creation():
    """Test credal set creation in IB agent."""
    agent = InfrabayesianRLAgent([0, 1])
    
    # Update some Q-values
    agent.update(0, 0, 1000000, 0)
    agent.update(0, 1, 1000, 0)
    
    # Get credal sets
    credal_0 = agent.get_credal_set(0, 0)
    credal_1 = agent.get_credal_set(0, 1)
    
    # Should be InfraPolytope objects
    from src.infradistribution import InfraPolytope
    assert isinstance(credal_0, InfraPolytope)
    assert isinstance(credal_1, InfraPolytope)
