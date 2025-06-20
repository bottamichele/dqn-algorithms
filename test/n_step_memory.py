import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium.wrappers import NumpyToTorch

from torch.nn.functional import mse_loss
from torch.optim import Adam

from dqn_algorithms.nn import DuelingDQN
from dqn_algorithms.memory import NStepUniformMemory, NStepPropPriorMemory
from dqn_algorithms.policy import e_greedy_policy
from dqn_algorithms.q_methods import compute_q_ddqn, compute_q_dqn

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

EPISODES = 250
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_INIT = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 10**-3
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

    #Create model and target model.
    model = DuelingDQN(OBSERVATION_SIZE, ACTION_SIZE, 256, 1, 1, 1).to(DEVICE)
    target_model = DuelingDQN(OBSERVATION_SIZE, ACTION_SIZE, 256, 1, 1, 1).to(DEVICE)
    target_model.load_state_dict(model.state_dict())

    model.train()

    #Define optimizer.
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #Create memory replay.
    if not USE_PRIORITY_MEMORY:
        memory = NStepUniformMemory(MEMORY_SIZE, N_STEP, GAMMA, OBSERVATION_SIZE, device=DEVICE)
    else:
        memory = NStepPropPriorMemory(MEMORY_SIZE, N_STEP, GAMMA, OBSERVATION_SIZE, device=DEVICE)
        beta_decay = (1.0 - memory.beta) / EPISODES

    #Training phase.
    scores = []
    total_states = 0
    epsilon = EPSILON_INIT

    for episode in range(1, EPISODES+1):
        #Episode.
        obs, _ = env.reset()
        epis_done = False
        scores.append(0)
        
        while not epis_done:
            #Choose action.
            action = e_greedy_policy(model,
                                     epsilon,
                                     obs)

            #Perform action choosen.
            next_obs, reward, terminated, truncated, _ = env.step(action)
            epis_done = terminated or truncated

            #Store current transiction.
            memory.store_transiction(obs, action, reward, next_obs, epis_done)

            #Do train step.
            if len(memory) >= BATCH_SIZE:
                #Sample mini-batch.
                if not USE_PRIORITY_MEMORY:
                    obs_b, action_b, reward_b, next_obs_b, next_obs_done_b = memory.sample_batch(BATCH_SIZE)
                else:
                    obs_b, action_b, reward_b, next_obs_b, next_obs_done_b, weight_b = memory.sample_batch(BATCH_SIZE)

                #Convertion.
                action_b = action_b.to(dtype=tc.int32)
                next_obs_done_b = next_obs_done_b.to(dtype=tc.int32)

                #Computet q values.
                q, q_target = compute_q_ddqn(model, target_model, GAMMA, obs_b, action_b, reward_b, next_obs_b, next_obs_done_b)

                #Compute loss and gradient.
                if not USE_PRIORITY_MEMORY:
                    loss = mse_loss(q, q_target).to(DEVICE)
                else:
                    loss = tc.mean((q_target - q).pow(2) * weight_b)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Update priorities if USE_PRIORITY_MEMORY is True.
                if USE_PRIORITY_MEMORY:
                    td_errors = tc.clamp(q_target - q, -1.0, 1.0)
                    memory.update_priorities(td_errors)
                    
            #Updates.
            obs = next_obs
            epsilon = epsilon - EPSILON_DECAY if epsilon > EPSILON_MIN else EPSILON_MIN
            scores[-1] += reward
            total_states += 1

            if total_states % UPDATE_RATE == 0:
                target_model.load_state_dict(model.state_dict())

        #Print training stats of current epidode ended.
        print("- Episode {:3d}: score = {:3.0f}; avg score = {:3.2f}; total states = {:>5d}; epsilon = {:.2f}".format(episode, scores[-1], np.mean(scores[-100:]), total_states, epsilon))

        if USE_PRIORITY_MEMORY:
            memory.beta += beta_decay

    env.close()