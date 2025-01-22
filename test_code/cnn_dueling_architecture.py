import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium import Wrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation

from torch.nn.functional import mse_loss
from torch.optim import Adam

from dqn.cnn.cnn_dqn import CnnDQN
from dqn.cnn.cnn_dueling_dqn import CnnDuelingDQN
from dqn.memory.uniform_memory import UniformMemory
from dqn.policy import e_greedy_policy
from dqn.q_methods import compute_q_ddqn

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

FRAME_SKIP = 4
TARGET_TOTAL_FRAMES = 100000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_INIT = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 10**-4
UPDATE_RATE = 250
MEMORY_SIZE = 10000
LEARNING_RATE = 10**-4
DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")
USE_DUELING_ARCHITECTURE = False

# ========================================
# =============== COSTANTS ===============
# ========================================

OBSERVATION_SIZE = 0
ACTION_SIZE = 0

# ========================================
# ======== FRAME SKIP OBSERVATION ========
# ========================================

class FrameSkipObservation(Wrapper):
    def __init__(self, env, frame_skip):
        super().__init__(env)
        self._frame_skip = frame_skip

    def step(self, action):
        total_reward = 0.0
        
        for _ in range(self._frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

# ========================================
# ================= MAIN =================
# ========================================

def main():
    global OBSERVATION_SIZE
    global ACTION_SIZE

    #Create enviroment.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = TransformObservation(env, lambda x:env.render(), Box(low=0, high=255, shape=(400, 600, 3), dtype=np.uint8))
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4, padding_type="zero")
    OBSERVATION_SIZE = env.observation_space.shape
    ACTION_SIZE = env.action_space.n

    #Create model and target model.
    if not USE_DUELING_ARCHITECTURE:
        model = CnnDQN(obs_size=OBSERVATION_SIZE, action_size=ACTION_SIZE).to(DEVICE)
        target_model = CnnDQN(obs_size=OBSERVATION_SIZE, action_size=ACTION_SIZE).to(DEVICE)
    else:
        model = CnnDuelingDQN(obs_size=OBSERVATION_SIZE, action_size=ACTION_SIZE).to(DEVICE)
        target_model = CnnDuelingDQN(obs_size=OBSERVATION_SIZE, action_size=ACTION_SIZE).to(DEVICE)
    target_model.load_state_dict(model.state_dict())

    model.train()

    #Define optimizer.
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #Create memory replay.
    memory = UniformMemory(MEMORY_SIZE, OBSERVATION_SIZE, obs_dtype=np.uint8)

    #Training phase.
    scores = []
    total_frames = 0
    episode = 0
    epsilon = EPSILON_INIT

    while total_frames <= TARGET_TOTAL_FRAMES:
        #Episode.
        obs, _ = env.reset()
        episode += 1
        epis_done = False
        scores.append(0)
        
        while not epis_done:
            #Choose action.
            action = e_greedy_policy(model,
                                     epsilon,
                                     tc.Tensor(np.array([obs])).to(DEVICE) / 255.0)

            #Perform action choosen.
            next_obs, reward, terminated, truncated, _ = env.step(action)
            epis_done = terminated or truncated

            #Store current transiction.
            memory.store_transiction(obs, action, reward, next_obs, epis_done)

            #Do train step.
            if len(memory) >= BATCH_SIZE:
                #Sample mini-batch.
                obs_b, action_b, reward_b, next_obs_b, next_obs_done_b = memory.sample_batch(BATCH_SIZE)

                #Convert to tensor.
                obs_b           = tc.Tensor(obs_b).to(DEVICE, dtype=tc.float32) / 255.0
                action_b        = tc.Tensor(action_b).to(DEVICE, dtype=tc.int32)
                reward_b        = tc.Tensor(reward_b).to(DEVICE, dtype=tc.float32)
                next_obs_b      = tc.Tensor(next_obs_b).to(DEVICE, dtype=tc.float32) / 255.0
                next_obs_done_b = tc.Tensor(next_obs_done_b).to(DEVICE, dtype=tc.int32)

                #Computet q values.
                q, q_target = compute_q_ddqn(model, target_model, GAMMA, obs_b, action_b, reward_b, next_obs_b, next_obs_done_b)

                #Compute loss and gradient.
                loss = mse_loss(q, q_target).to(DEVICE)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #Updates.
            obs = next_obs
            epsilon = epsilon - EPSILON_DECAY if epsilon > EPSILON_MIN else EPSILON_MIN
            scores[-1] += reward
            total_frames += 1

            if total_frames % UPDATE_RATE == 0:
                target_model.load_state_dict(model.state_dict())

        #Print training stats of current epidode ended.
        print("- Episode {:3d}: score = {:3d}; avg score = {:3.2f}; total frames = {:>5d}; epsilon = {:.2f}".format(int(episode), int(scores[-1]), np.mean(scores[-100:]), int(total_frames), epsilon))

    env.close()