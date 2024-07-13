from utils import AbstractLogged
from typing import Any
import json
import pickle


class DataLoader(AbstractLogged):
    '''
    Loads and saves data for offline reinforcement learning.

    data: list of lists of tuples (state, action, reward, info), where each list is the trajectory of a single episode. info is a dictionary containing additional information. reward is a scalar. per convention, the last action in the trajectory is None and the last reward is 0.0.
    '''
    data: list[list[tuple[Any, Any, float, dict[str, Any]]]]

    def __init__(self, data_path: str):
        '''
        Initializes the data loader.
        
        Args:
            data_path: The path to the data file.
        '''
        super().__init__()
        self.data_path = data_path
        self.data = []
        self.load_data()

    def save_data(self):
        '''
        Saves the data in self.data to json file.
        '''
        self.log(f"Saving data to {self.data_path}")
        with open(self.data_path, 'w') as f:
            json.dump(self.data, f)

    def load_data(self):
        '''
        Loads the data from the json file at self.data_path.
        '''
        self.log(f"Loading data from {self.data_path}")
        try:
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.log(f"Data file not found at {self.data_path}. Initializing empty data.")
            self.data = []

    def add_trajectory(self, traj: list[tuple[Any, Any, float, dict[str, Any]]]):
        '''
        Adds a trajectory to the data.

        Args:
            traj: The trajectory to add.
        '''
        self.data.append(traj)

    def get_flattened_dataset(self):
        '''
        Returns a flattened version of the dataset.

        Returns:
            A list of tuples (state, action, reward, info).
        '''
        return [item for traj in self.data for item in traj]
    
    def get_flatted_dataset_filtered_by_action(self, action: Any):
        '''
        Returns a flattened version of the dataset, filtered by action.

        Args:
            action: The action to filter by.

        Returns:
            A list of tuples (state, action, reward, info) where the action is equal to the input action.
        '''
        return [item for traj in self.data for item in traj if item[1] == action]
    
    def __repr__(self):
        return f"DataLoader(data_path={self.data_path}, data={self.data.__repr__()})"
    
    def add_default_info(self):
        '''
        Adds the default info to all trajectories.
        Default info includes:
        - 'episode_return': The return of the episode.
        - 'episode_length': The length of the episode.
        - 'future_return': The future return starting from the state in that episode.
        - 'next_state': The next state in the episode.
        '''

        for traj in self.data:
            episode_return = sum([item[2] for item in traj])
            episode_length = len(traj)
            future_return = 0

            # loop backwards to calculate future return
            for i, item in enumerate(reversed(traj)):
                future_return += item[2]
                item[3]['episode_return'] = episode_return
                item[3]['episode_length'] = episode_length
                item[3]['future_return'] = future_return
                if i < episode_length - 1:
                    item[3]['next_state'] = traj[i + 1][0]
                else:
                    item[3]['next_state'] = None

    def filter_top_trajectories(self, n: int):
        '''
        Filters the top n trajectories by episode return.

        Args:
            n: The number of top trajectories to keep.
        '''

        # first add default info
        # self.add_default_info()

        # sort by episode return (item[3] is the info dictionary)
        self.data.sort(key=lambda traj: traj[0][3]['episode_return'], reverse=True)

        # keep only the top n trajectories
        self.data = self.data[:n]
        self.log(f"Filtered top {n} trajectories by episode return.")

    def filter_top_trajectories_fraction(self, p: float):
        '''
        Filters the top p fraction of trajectories by episode return.

        Args:
            p: The fraction of top trajectories to keep.
        '''

        # first add default info
        # self.add_default_info()

        # sort by episode return (item[3] is the info dictionary)
        self.data.sort(key=lambda traj: traj[0][3]['episode_return'], reverse=True)

        # keep only the top p fraction of trajectories
        print(len(self.data))
        print(p)
        first_k = int(p * len(self.data))
        self.data = self.data[:first_k]
        self.log(f"Filtered top {p} fraction of trajectories by episode return.")
        
class DataLoaderPickle(DataLoader):
    '''
    Loads and saves to pickle instead
    '''
    def save_data(self):
        '''
        Saves the data in self.data to pickle file.
        '''
        self.log(f"Saving data to {self.data_path}")
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.data, f)

    def load_data(self):
        '''
        Loads the data from the pickle file at self.data_path.
        '''
        self.log(f"Loading data from {self.data_path}")
        try:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            self.log(f"Data file not found at {self.data_path}. Initializing empty data.")
            self.data = []
