import numpy as np

def compute_q_dqn(current_model, gamma, obs, action, reward, next_obs, next_obs_done, target_model=None):
    """Compute q values using Deep Q-Networks method
    
    Parameters
    --------------------
    current_model: DQN-like
        current model
        
    gamma: float
        discount factor
        
    obs: tc.Tensor
        observations
        
    action: tc.Tensor
        actions performed on obs
        
    reward: tc.Tensor
        rewards obtained performing action
    
    next_obs: tc.Tensor
        next observations of obs
    
    next_obs_done: tc.Tensor
        next observations if they are terminal states or not
    
    target_model: DQN-like, optional
        target model
        
    Returns
    --------------------
    q: tc.Tensor
        q values of current model
        
    q_target: tc.Tensor
        q target values"""
    
    assert(obs.shape[0] == action.shape[0] and obs.shape[0] == reward.shape[0] and obs.shape[0] == next_obs.shape[0] and obs.shape[0] == next_obs_done.shape[0])
    idxs = np.arange(0, obs.shape[0], 1, dtype=np.int32)
    
    if target_model is None:
        target_model = current_model

    q = current_model(obs)[idxs, action]
    q_target = reward + (1 - next_obs_done) * gamma * target_model(next_obs).amax(dim=1)

    return q, q_target

def compute_q_ddqn(current_model, target_model, gamma, obs, action, reward, next_obs, next_obs_done):
    """Compute q values using Double Deep Q-Networks method.
    
    Parameters
    --------------------
    current_model: DQN-like
        current model

    target_model: DQN-like
        target model
        
    gamma: float
        discount factor
        
    obs: tc.Tensor
        observations
        
    action: tc.Tensor
        actions performed on obs
        
    reward: tc.Tensor
        rewards obtained performing action
    
    next_obs: tc.Tensor
        next observations of obs
    
    next_obs_done: tc.Tensor
        next observations if they are terminal states or not
        
    Retunrns
    --------------------
    q: tc.Tensor
        q values of current model
        
    q_target: tc.Tensor
        q target values"""
    
    assert(obs.shape[0] == action.shape[0] and obs.shape[0] == reward.shape[0] and obs.shape[0] == next_obs.shape[0] and obs.shape[0] == next_obs_done.shape[0])
    idxs = np.arange(0, obs.shape[0], 1, dtype=np.int32)

    q = current_model(obs)[idxs, action]

    best_act_next = current_model(next_obs).argmax(dim=1)
    q_target = reward + (1 - next_obs_done) * gamma * target_model(next_obs)[idxs, best_act_next]

    return q, q_target