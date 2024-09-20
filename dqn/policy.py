import numpy as np

def e_greedy_policy(model, epsilon, obs):
    """Choose an action to perform using e-greedy policy.
    
    Parameters
    --------------------
    model: DQN-like
        a DQN model
        
    epsilon: float
        epsilon value for e-greedy policy
        
    obs: tc.Tensor
        an observation of enviroment
        
    Return
    --------------------
    action: int
        action choosen to perform"""
    
    rng = np.random.default_rng()
    q = model(obs)

    if rng.uniform() <= epsilon:
        return rng.integers(0, q.shape[1])
    else:
        return q.argmax(dim=1).item()

def greedy_policy(model, obs):
    """Choose an action to perform using greedy policy.
    
    Parameters
    --------------------
    model: DQN-like
        a DQN model
        
    obs: tc.Tensor
        an observation of enviroment
        
    Return
    --------------------
    action: int
        action choosen to perform"""

    return model(obs).argmax(dim=1).item()