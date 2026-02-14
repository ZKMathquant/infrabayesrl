import torch as th
from typing import Tuple, Dict, List
from collections import defaultdict

class NewcombEnvironment:
    """Newcomb's paradox as policy-dependent RL environment."""
    
    def __init__(self, predictor_accuracy: float = 0.9):
        self.predictor_accuracy = predictor_accuracy
        self.state = 0
        self.actions = [0, 1]  # 0: one-box, 1: two-box
        self.agent_history = []
        self.episode_count = 0
        self.prediction_history = []
    
    def reset(self) -> int:
        """Reset environment for new episode."""
        self.episode_count += 1
        return self.state
    
    def predict_agent_policy(self) -> int:
        """Predictor estimates agent's likely choice."""
        if len(self.agent_history) < 5:
            return th.randint(0, 2, (1,)).item()
        
        # Analyze recent behavior
        recent = self.agent_history[-20:]
        one_box_rate = sum(1 for a in recent if a == 0) / len(recent)
        
        # Predictor accuracy affects prediction quality
        if th.rand(1).item() < self.predictor_accuracy:
            # Correct prediction based on observed pattern
            return 0 if one_box_rate > 0.5 else 1
        else:
            # Incorrect prediction
            return 1 if one_box_rate > 0.5 else 0
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """Execute action in policy-dependent environment."""
        self.agent_history.append(action)
        
        # Predictor's decision (made before agent acts)
        predicted_action = self.predict_agent_policy()
        self.prediction_history.append(predicted_action)
        
        # Newcomb rewards
        if action == 0:  # One-box
            reward = 1000000.0 if predicted_action == 0 else 0.0
        else:  # Two-box
            reward = 1000.0 + (1000000.0 if predicted_action == 0 else 0.0)
        
        return self.state, reward, True, {
            'predicted': predicted_action,
            'actual': action,
            'episode': self.episode_count,
            'predictor_accuracy': self.get_predictor_accuracy()
        }
    
    def get_predictor_accuracy(self) -> float:
        """Calculate actual predictor accuracy."""
        if len(self.agent_history) < 2:
            return 0.5
        
        correct = 0
        for i in range(1, len(self.agent_history)):
            if self.prediction_history[i-1] == self.agent_history[i]:
                correct += 1
        
        return correct / (len(self.agent_history) - 1)

class LogicalPredictorEnv(NewcombEnvironment):
    """Enhanced predictor analyzing agent's decision algorithm."""
    
    def __init__(self, predictor_accuracy: float = 0.95):
        super().__init__(predictor_accuracy)
        self.consistency_tracker = defaultdict(int)
        self.pattern_memory = []
    
    def predict_agent_policy(self) -> int:
        """Sophisticated logical predictor."""
        if len(self.agent_history) < 10:
            return super().predict_agent_policy()
        
        # Analyze consistency and patterns
        recent = self.agent_history[-50:]
        
        # Check for perfect consistency
        if len(set(recent)) == 1:
            # Agent is perfectly consistent
            if th.rand(1).item() < 0.98:  # Very high accuracy
                return recent[0]
        
        # Check for alternating patterns
        if len(recent) >= 4:
            alternating = all(recent[i] != recent[i+1] for i in range(len(recent)-1))
            if alternating and th.rand(1).item() < 0.9:
                return 1 - recent[-1]  # Predict opposite of last
        
        # Fall back to base predictor
        return super().predict_agent_policy()

class MultiPredictorEnv(NewcombEnvironment):
    """Environment with multiple predictors of varying accuracy."""
    
    def __init__(self, predictor_accuracies: List[float] = [0.8, 0.9, 0.95]):
        super().__init__(predictor_accuracies[0])
        self.predictors = predictor_accuracies
        self.current_predictor = 0
    
    def reset(self) -> int:
        """Reset and potentially switch predictor."""
        if self.episode_count % 100 == 0:
            self.current_predictor = (self.current_predictor + 1) % len(self.predictors)
            self.predictor_accuracy = self.predictors[self.current_predictor]
        
        return super().reset()
