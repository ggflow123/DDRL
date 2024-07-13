import torch
import numpy as np
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from wrappers import Teacher
import os
from ppo import PPO as ProcgenPPO
from policies import ImpalaCNN
from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize

class PPOTeacherCartpole(Teacher):
    '''
    Wrapper for teacher model using Stable Baselines3 PPO
    '''

    def __init__(self, model_name="PPO-Teacher-Cartpole"):
        self.model_name = model_name  # Name of the model used to train the student

        checkpoint = load_from_hub( repo_id="sb3/ppo-CartPole-v1", filename="ppo-CartPole-v1.zip",)
        self.model = PPO.load(checkpoint, print_system_info=True)
        #self.model = PPO("MlpPolicy", env, verbose=1)  # Initialize the PPO model with MlpPolicy

    def get_knowledge(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Returns the knowledge of the teacher that is used to train the student.
        In this case, it returns the action probabilities.
        '''
        action, _states = self.model.predict(state, deterministic=False)
        # this action is an integer. we need to convert it to a one-hot vector
        return torch.tensor(action)

    def step(self, state: torch.Tensor):
        '''
        Returns the next action given the current state.
        '''
        action, _states = self.model.predict(state, deterministic=True)
        return action

    def train(self, total_timesteps):
        '''
        Train the PPO model
        '''
        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self, path):
        '''
        Save the model to a file
        '''
        self.model.save(path)

    def load_model(self, path):
        '''
        Load the model from a file
        '''
        self.model = PPO.load(path)

class PPOTeacherHeist(Teacher):
    '''
    Wrapper for teacher model using Stable Baselines3 PPO
    '''

    def __init__(self, model_name="PPO-Teacher-Cartpole", action_space_length=15, device='auto'):
        self.model_name = model_name  # Name of the model used to train the student
        self.action_space_length = action_space_length # Number of actions in the action space

        self.model = None
        #self.model = PPO("MlpPolicy", env, verbose=1)  # Initialize the PPO model with MlpPolicy



    def get_knowledge(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Returns the knowledge of the teacher that is used to train the student.
        In this case, it returns the action probabilities.
        '''
        state = np.array(state.cpu())
        action, _states = self.model.predict(state, deterministic=False)
        # this action is an integer. we need to convert it to a one-hot vector
        action = np.eye(self.action_space_length)[action]
        return torch.tensor(action)

    def step(self, state: torch.Tensor):
        '''
        Returns the next action given the current state.
        '''
        state = np.array(state.cpu())
        action, _states = self.model.predict(state, deterministic=True)
        #print(f"inside teacher step: {action}, {state}, {_states}")
        return action

    def predict(self, observations, state, episode_start, deterministic):
        return self.model.predict(observations, state=state, episode_start=episode_start, deterministic=deterministic,
            )
    def train(self, total_timesteps):
        '''
        Train the PPO model
        '''
        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self, path):
        '''
        Save the model to a file
        '''

        self.model.save(path)

    def load_model(self, path):
        '''
        Load the model from a file
        '''
        print(os.getcwd())
        self.model = PPO.load(path)


class PPOTeacherProcgen(Teacher):
    '''
    Wrapper for teacher model using Stable Baselines3 PPO
    '''

    def __init__(self, model_name="PPO-Teacher-Procgen", action_space_length=15, device='auto'):
        self.model_name = model_name  # Name of the model used to train the student
        self.action_space_length = action_space_length # Number of actions in the action space

        self.model = None
        #self.model = ProcgenPPO("MlpPolicy", env, verbose=1)  # Initialize the PPO model with MlpPolicy



    def get_knowledge(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Returns the knowledge of the teacher that is used to train the student.
        In this case, it returns the action probabilities.
        '''
        #state = torch.unsqueeze(state, dim=0)
        state = np.array(state.cpu())
        action = self.model.batch_act(state)
        #action = action.squeeze()
        # this action is an integer. we need to convert it to a one-hot vector
        action = np.eye(self.action_space_length)[action]
        #torch.softmax(self.base_model.forward(state.to(self.device)), dim=-1)
        return action

    def get_knowledge_distill_train(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Returns the knowledge of the teacher that is used to train the student.
        In this case, it returns the action probabilities.
        '''
        #state = torch.unsqueeze(state, dim=0)
        #state = np.array(state.cpu())
        action = self.model.batch_act_distill_train(state)
        #action = action.squeeze()
        # this action is an integer. we need to convert it to a one-hot vector
        #action = np.eye(self.action_space_length)[action]
        #torch.softmax(self.base_model.forward(state.to(self.device)), dim=-1)
        return action.probs
    def step(self, state: torch.Tensor):
        '''
        Returns the next action given the current state.
        '''
        #state = torch.unsqueeze(state, dim=0)
        #state = np.array(state.cpu())
        #print(state.shape)
        action = self.model.batch_act(state)
        #action = action.squeeze()
        #print(f"inside teacher step: {action}, {state}, {_states}")
        return action


    def save_model(self, path):
        '''
        Save the model to a file
        '''

        self.model.model.save_to_file(model_path)

    def load_model(self, path):
        '''
        Load the model from a file
        '''
        print(os.getcwd())
        # The below env initialization is only for policy network init.
        venv = ProcgenEnv(
            num_envs=1,
            env_name='starpilot',
            num_levels=0,
            start_level=0,
            distribution_mode='easy',
            num_threads=1,
        )
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        # Create policy.
        policy = ImpalaCNN(
            obs_space=venv.observation_space,
            num_outputs=venv.action_space.n,
        )
        state_dict = torch.load(path)
        policy.load_state_dict(state_dict)
        #print(policy.keys())
        # Create Model and load Model
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)
        self.model = ProcgenPPO(
            model=policy,
            optimizer=optimizer,
            gpu=0,
            gamma=0.999,
            lambd=0.95,
            value_func_coef=0.5,
            entropy_coef=0.1,
            update_interval= 256 * 64,
            minibatch_size=8,
            epochs=3,
            clip_eps=0.2,
            clip_eps_vf=0.2,
            max_grad_norm=0.5,
        )
