import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium.wrappers import FrameStackObservation, FlattenObservation

from torch.nn.functional import mse_loss
from torch.optim import Adam

from dqn.nn.dqn import DQN
from dqn.nn.dueling_dqn import DuelingDQN
from dqn.memory.prop_prio_mem import PropPriorMemory
from dqn.memory.uniform_memory import UniformMemory
from dqn.policy import e_greedy_policy
from dqn.q_methods import compute_q_ddqn, compute_q_dqn

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

LAST_N_STATES = 4
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_INIT = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 10**-4
UPDATE_RATE = 250
MEMORY_SIZE = 5000
LEARNING_RATE = 10**-3
DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")
USE_DDQN = True
USE_PRIORITY_MEMORY = True
USE_DUELING_ARCHITECTURE = True

# ========================================
# =============== COSTANTS ===============
# ========================================

OBSERVATION_SIZE = 0
ACTION_SIZE = 0
RNG = np.random.default_rng()

# ========================================
# ================= MAIN =================
# ========================================

def main():
    global OBSERVATION_SIZE
    global ACTION_SIZE

    #Create enviroment.
    env = gym.make("CartPole-v1")
    if LAST_N_STATES >= 2:
        env = FrameStackObservation(env, stack_size=LAST_N_STATES, padding_type="zero")
        env = FlattenObservation(env)
    OBSERVATION_SIZE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n

    #Create model and target model.
    if not USE_DUELING_ARCHITECTURE:
        model = DQN(OBSERVATION_SIZE, ACTION_SIZE, 256, 2).to(DEVICE)
        target_model = DQN(OBSERVATION_SIZE, ACTION_SIZE, 256, 2).to(DEVICE)
    else:
        model = DuelingDQN(OBSERVATION_SIZE, ACTION_SIZE, 256, 1, 1, 1).to(DEVICE)
        target_model = DuelingDQN(OBSERVATION_SIZE, ACTION_SIZE, 256, 1, 1, 1).to(DEVICE)
    target_model.load_state_dict(model.state_dict())

    model.train()

    #Define optimizer.
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #Create memory replay.
    if not USE_PRIORITY_MEMORY:
        memory = UniformMemory(MEMORY_SIZE, OBSERVATION_SIZE)
    else:
        memory = PropPriorMemory(MEMORY_SIZE, OBSERVATION_SIZE)
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
                                     tc.Tensor(np.array([obs])).to(DEVICE))

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

                #Convert to tensor.
                obs_b           = tc.Tensor(obs_b).to(DEVICE, dtype=tc.float32)
                action_b        = tc.Tensor(action_b).to(DEVICE, dtype=tc.int32)
                reward_b        = tc.Tensor(reward_b).to(DEVICE, dtype=tc.float32)
                next_obs_b      = tc.Tensor(next_obs_b).to(DEVICE, dtype=tc.float32)
                next_obs_done_b = tc.Tensor(next_obs_done_b).to(DEVICE, dtype=tc.int32)

                if USE_PRIORITY_MEMORY:
                    weight_b = tc.Tensor(weight_b).to(DEVICE, dtype=tc.float32)

                #Computet q values.
                if not USE_DDQN:
                    q, q_target = compute_q_dqn(model, GAMMA, obs_b, action_b, reward_b, next_obs_b, next_obs_done_b, target_model)
                else:
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
                    memory.update_priorities(td_errors.cpu().detach().numpy())
                    memory.beta += beta_decay

            #Updates.
            obs = next_obs
            epsilon = epsilon - EPSILON_DECAY if epsilon > EPSILON_MIN else EPSILON_MIN
            scores[-1] += reward
            total_states += 1

            if total_states % UPDATE_RATE == 0:
                target_model.load_state_dict(model.state_dict())

        #Print training stats of current epidode ended.
        print("- Episode {:3d}: score = {:3d}; avg score = {:3.2f}; total states = {:>5d}; epsilon = {:.2f}".format(int(episode), int(scores[-1]), np.mean(scores[-100:]), int(total_states), epsilon))

    env.close()