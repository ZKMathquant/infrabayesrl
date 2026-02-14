import torch as th
import torch.distributions as thd
from typing import Dict, List, Tuple
from collections import defaultdict
from .infradistribution import InfraPolytope
from .sa_measure import SaMeasure

class InfrabayesianRLAgent:
    """RL agent using infrabayesian epistemology."""
    
    def __init__(self, actions: List[int], uncertainty_radius: float = 0.1, 
                 learning_rate: float = 0.1, epsilon: float = 0.05, gamma: float = 0.9):
        self.actions = actions
        self.uncertainty_radius = uncertainty_radius
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Q-values and visit counts
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        self.q_history = defaultdict(lambda: defaultdict(list))
        
    def get_credal_set(self, state: int, action: int) -> InfraPolytope:
        """Create infradistribution over Q-values."""
        base_q = self.q_values[state][action]
        visits = max(1, self.visit_counts[state][action])
        
        # Uncertainty decreases with visits
        radius = self.uncertainty_radius / th.sqrt(th.tensor(visits, dtype=th.float32))
        
        # Create credal set with multiple Q-value estimates
        q_estimates = th.tensor([
            base_q - radius,
            base_q,
            base_q + radius
        ])
        
        # Create batch of Normal distributions instead of Categorical
        batch_normal = thd.Normal(q_estimates, th.ones(3) * 0.01)
        return InfraPolytope(batch_normal)
    
    def infrabayesian_value(self, state: int) -> float:
        """Compute infrabayesian value using min over credal sets."""
        if not self.actions:
            return 0.0
        
        action_values = []
        for action in self.actions:
            credal_set = self.get_credal_set(state, action)
            # Min expected value (pessimistic approach)
            min_value = credal_set(lambda x: x).item()
            action_values.append(min_value)
        
        return max(action_values)
    
    def select_action(self, state: int) -> int:
        """Select action using infrabayesian decision rule."""
        if th.rand(1).item() < self.epsilon:
            return th.randint(0, len(self.actions), (1,)).item()
        
        # Compute infrabayesian Q-values
        best_action = self.actions[0]
        best_value = float('-inf')
        
        for action in self.actions:
            credal_set = self.get_credal_set(state, action)
            value = credal_set(lambda x: x).item()
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """Update using infrabayesian Bellman equation."""
        self.visit_counts[state][action] += 1
        
        # Compute target using infrabayesian value
        next_value = self.infrabayesian_value(next_state)
        target = reward + self.gamma * next_value
        
        # Update Q-value
        current_q = self.q_values[state][action]
        self.q_values[state][action] += self.learning_rate * (target - current_q)
        
        # Store for uncertainty estimation
        self.q_history[state][action].append(target)
        if len(self.q_history[state][action]) > 100:
            self.q_history[state][action] = self.q_history[state][action][-100:]

class ClassicalRLAgent:
    """Standard Q-learning agent for comparison."""
    
    def __init__(self, actions: List[int], learning_rate: float = 0.1, 
                 epsilon: float = 0.05, gamma: float = 0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_values = defaultdict(lambda: defaultdict(float))
    
    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if th.rand(1).item() < self.epsilon:
            return th.randint(0, len(self.actions), (1,)).item()
        
        best_action = max(self.actions, key=lambda a: self.q_values[state][a])
        return best_action
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """Standard Q-learning update."""
        next_value = max(self.q_values[next_state][a] for a in self.actions) if self.actions else 0
        target = reward + self.gamma * next_value
        current_q = self.q_values[state][action]
        self.q_values[state][action] += self.learning_rate * (target - current_q)
