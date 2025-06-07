import os
import copy
import pickle
import torch as tc

from torch.optim import Adam
from torch.nn.functional import mse_loss

from dqn_algorithms.memory import UniformMemory, PropPriorMemory, NStepUniformMemory, NStepPropPriorMemory
from dqn_algorithms.policy import e_greedy_policy
from dqn_algorithms.q_methods import compute_q_dqn, compute_q_ddqn

class DQNAgent:
    """A DQN training agent."""

    def __init__(self, model, mem_params, update_target_step, batch_size, lr, gamma, eps_init, eps_fin, eps_dec, device, use_double=True):
        """Create new DQN training agent.
        
        Parameters
        --------------------
        model: DQN-like
            a DQN model

        mem_params: dict
            memory replay parameters. The dict contains the type of memory replay and
            its own parameters.

        update_target_step: int
            how many steps target model is updated
            
        batch_size: int
            minibatch size

        lr: float
            learning rate

        gamma: float
            discount factor

        eps_init: float
            initial epsilon value

        eps_fin: float
            final epsilon value

        eps_dec: float
            epsilon decay 

        device: tc.device
            device model is trained on

        use_double: bool, optional
            whether or not to use Double Q-Learning
        """

        #Hyperparameters.
        self.update_target_step = update_target_step
        self.use_double = use_double
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = eps_init
        self.epsilon_final = eps_fin
        self.epsilon_decay = eps_dec
        self.device = device

        #DQN model.
        self.model = model.to(device=device)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.model.train()
        self.target_model.eval()

        #Memory replay.
        if mem_params["type"] == "uniform_memory":
            self.memory = UniformMemory(mem_params["mem_size"], 
                                        mem_params["obs_size"], 
                                        mem_params.get("obs_dtype", tc.float32), 
                                        tc.int32, 
                                        tc.float32, 
                                        device=device)
            self._use_per = False
        elif mem_params["type"] == "proportional_prioritized_memory":
            self.memory = PropPriorMemory(mem_params["mem_size"], 
                                          mem_params["obs_size"],
                                          alpha=mem_params.get("alpha", 0.6),
                                          beta=mem_params.get("beta", 0.4),
                                          eps=mem_params.get("eps", 10**-5),
                                          obs_dtype=mem_params.get("obs_dtype", tc.float32),
                                          act_dtype=tc.int32,
                                          rew_dtype=tc.float32,
                                          device=device)
            self._use_per = True
        elif mem_params["type"] == "n_step_uniform_memory":
            self.memory = NStepUniformMemory(mem_params["mem_size"],
                                             mem_params["n_step"],
                                             gamma, 
                                             mem_params["obs_size"], 
                                             mem_params.get("obs_dtype", tc.float32), 
                                             tc.int32, 
                                             tc.float32, 
                                             device=device)
            self._use_per = False
        elif mem_params["type"] == "n_step_proportional_prioritized_memory":
            self.memory = NStepPropPriorMemory(mem_params["mem_size"],
                                               mem_params["n_step"],
                                               gamma,
                                               mem_params["obs_size"],
                                               alpha=mem_params.get("alpha", 0.6),
                                               beta=mem_params.get("beta", 0.4),
                                               eps=mem_params.get("eps", 10**-5),
                                               obs_dtype=mem_params.get("obs_dtype", tc.float32),
                                               act_dtype=tc.int32,
                                               rew_dtype=tc.float32,
                                               device=device)
            self._use_per = True
        else:
            raise ValueError("The dict \'mem_params\' does not specify any known type of memory replay.")

    def choose_action(self, obs):
        """Choose an action to perform.
        
        Parameter
        --------------------
        obs: tc.Tensor
            an observation"""
        
        return e_greedy_policy(self.model, self.epsilon, obs.to(device=self.device))
    
    def remember(self, obs, action, reward, next_obs, next_obs_done):
        """Remember a transiction.
        
        Parameters
        --------------------
        obs: tc.Tensor
            an observation
            
        action: int
            an action choosen from obs
            
        reward: float
            reward obtained to perform action
        
        next_obs: tc.Tensor
            obsevation obtained to perform action
            
        next_obs_done: bool
            True if next_obs is a terminal state, False otherwise."""
        
        self.memory.store_transiction(obs, action, reward, next_obs, next_obs_done)

    def train(self):
        """Do a train step.
        
        Return
        --------------------
        train_infos: dict
            a dictionary wihch contains some information about the train step computed"""

        infos = {}

        if len(self.memory) >= self.batch_size:
            #Sample a mini-batch.
            if self._use_per:
                obs_b, action_b, reward_b, next_obs_b, next_obs_done_b, weight_b = self.memory.sample_batch(self.batch_size)
            else:
                obs_b, action_b, reward_b, next_obs_b, next_obs_done_b = self.memory.sample_batch(self.batch_size)

            next_obs_done_b = next_obs_done_b.to(dtype=tc.int32)

            #Computet q values.
            if not self.use_double:
                q, q_target = compute_q_dqn(self.model, self.gamma, obs_b, action_b, reward_b, next_obs_b, next_obs_done_b, self.target_model)
            else:
                q, q_target = compute_q_ddqn(self.model, self.target_model, self.gamma, obs_b, action_b, reward_b, next_obs_b, next_obs_done_b)

            #Compute loss and gradient.
            if self._use_per:
                loss = tc.mean((q_target - q).pow(2) * weight_b)
            else:
                loss = mse_loss(q, q_target).to(device=self.device)
            infos["loss"] = loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #Update priorities if memory replay is a PER.
            if self._use_per                                                                                                                                                                           :
                td_errors = tc.clamp(q_target - q, -1.0, 1.0)
                self.memory.update_priorities(td_errors)
                    
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final

        return infos

    def update_target_model(self):
        """Update target model's parameters."""

        self.target_model.load_state_dict(self.model.state_dict())

    def save_session(self, path):
        """Save current session of this agent on disk.
        
        Parameter
        --------------------
        path: str
            path where current session of this agent is saved on"""
        
        with open(os.path.join(path, "dqn_agent_session.pkl"), "wb") as session_file:
            pickle.dump(self, session_file)

    def load_session(path):
        """Load a session of DQN agent from disk.
        
        Parameter
        --------------------
        path where a session of DQN agent is on
        
        Return
        --------------------
        agent: DQNAgent
            a DQN training agent"""

        with open(os.path.join(path, "dqn_agent_session.pkl"), "rb") as session_file:
            agent = pickle.load(session_file)

        return agent

    def save_model(self, path):
        """Save model on disk.
        
        Parameter
        --------------------
        path: str
            path where model is saved on"""
        
        tc.save(self.model.state_dict(), os.path.join(path, "model.pth"))