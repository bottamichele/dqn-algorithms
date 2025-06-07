import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium.wrappers import NumpyToTorch

from dqn_algorithms.nn import DuelingDQN
from dqn_algorithms.agent import DQNAgent

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_INIT = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 10**-4
UPDATE_RATE = 250
MEMORY_SIZE = 5000
N_STEP = 3
LEARNING_RATE = 10**-3
DEVICE = tc.device("cpu")
USE_PRIORITY_MEMORY = True

# ========================================
# =============== COSTANTS ===============
# ========================================

OBSERVATION_SIZE = 0
ACTION_SIZE = 0

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    #Create enviroment.
    env = gym.make("CartPole-v1")
    env = NumpyToTorch(env, device=DEVICE)
    OBSERVATION_SIZE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n

    #Create model.
    model = DuelingDQN(OBSERVATION_SIZE, ACTION_SIZE, 256, 1, 1, 1).to(DEVICE)

    #Create memory replay params.
    if not USE_PRIORITY_MEMORY:
        type_memory = "uniform_memory" if N_STEP == 1 else "n_step_uniform_memory"
    else:
        type_memory = "proportional_prioritized_memory" if N_STEP == 1 else "n_step_proportional_prioritized_memory"
    
    memory_params = {"type": type_memory, "mem_size": MEMORY_SIZE, "obs_size": OBSERVATION_SIZE}
    if N_STEP > 1:
        memory_params.update({"n_step": N_STEP})
    
    #Define DQN training agent.
    agent = DQNAgent(model, memory_params, UPDATE_RATE, BATCH_SIZE, LEARNING_RATE, GAMMA, EPSILON_INIT, EPSILON_MIN, EPSILON_DECAY, DEVICE)

    #Training phase.
    scores = []
    total_states = 0

    for episode in range(1, EPISODES+1):
        #Episode.
        obs, _ = env.reset()
        epis_done = False
        scores.append(0)
        
        while not epis_done:
            #Choose action.
            action = agent.choose_action(obs)

            #Perform action choosen.
            next_obs, reward, terminated, truncated, _ = env.step(action)
            epis_done = terminated or truncated

            #Store current transiction.
            agent.remember(obs, action, reward, next_obs, epis_done)

            #Do train step.
            agent.train()
                    
            #Updates.
            obs = next_obs
            scores[-1] += reward
            total_states += 1

            if total_states % agent.update_target_step == 0:
                agent.update_target_model()

        #Print training stats of current epidode ended.
        print("- Episode {:3d}: score = {:3.0f}; avg score = {:3.2f}; total states = {:>5d}; epsilon = {:.2f}".format(episode, scores[-1], np.mean(scores[-100:]), total_states, agent.epsilon))

    env.close()