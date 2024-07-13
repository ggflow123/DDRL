import numpy as np
import math
import torch


class RolloutBuffer:

    def __init__(self, buffer_size=100, rng=None):
        self.buffer = []
        self.buffer_size = buffer_size
        self.rng = rng or np.random.default_rng()

    def add(self, observation):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(observation)

    def batch_add(self, observations):
        for observation in observations:
            self.add(observation)

    def sample(self, batch_size=1):
        if len(self.buffer) < batch_size:
            return self.buffer
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        #return self.rng.choice(self.buffer, size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def clear(self):
        self.buffer = []


class RolloutBufferBatch:
    def __init__(self, state_dim, knowledge_dim=None, buffer_size=100, device='cpu', rng=None):
        '''
        Buffer will have shape (buffer_size, state_dim) and (buffer_size, knowledge_dim)
        
        '''
        # Ensure dimensions for states and teacher-knowledge are provided
        #assert state_dim is not None "Dimensions must be specified"

        
        self.state_buffer = torch.empty((buffer_size, *state_dim), dtype=torch.float32, device=device)
        if knowledge_dim is not None:
            self.knowledge_buffer = torch.empty((buffer_size, *knowledge_dim), dtype=torch.float32, device=device)
        else:
            self.knowledge_buffer = torch.empty((buffer_size, 1), dtype=torch.float32, device=device)
        self.buffer_size = buffer_size
        self.buffer_count = 0
        self.device = device
        self.rng = rng or np.random.default_rng()

    def add(self, observation):
        state, teacher_knowledge = observation[0], observation[1]
        idx = self.buffer_count % self.buffer_size
        self.state_buffer[idx] = state.to(self.device) 
        self.knowledge_buffer[idx] = teacher_knowledge.to(self.device)
        self.buffer_count += 1

    def sample(self, batch_size=1):
        if self.buffer_count < batch_size:
            sample_indices = range(self.buffer_count)
        else:
            sample_indices = self.rng.choice(min(self.buffer_count, self.buffer_size), size=batch_size, replace=False)

        return (self.state_buffer[sample_indices], self.knowledge_buffer[sample_indices])

    def clear(self):
        self.buffer_count = 0
        self.state_buffer = torch.empty((buffer_size, *state_dim), dtype=torch.float32, device=device)
        if knowledge_dim is not None:
            self.knowledge_buffer = torch.empty((buffer_size, *knowledge_dim), dtype=torch.float32, device=device)
        else:
            self.knowledge_buffer = torch.empty(buffer_size, dtype=torch.float32, device=device)

    def to(self, device):
        self.state_buffer = self.state_buffer.to(device)
        self.knowledge_buffer = self.knowledge_buffer.to(device)
        self.device = device

    def save(self, file_path):
        data = {
            'state_buffer': self.state_buffer,
            'knowledge_buffer': self.knowledge_buffer,
            'buffer_size': self.buffer_size,
            'buffer_count': self.buffer_count,
            'device': self.device,
            'rng_state': self.rng.bit_generator.state
        }
        torch.save(data, file_path)
        print(f"Buffer saved to {file_path}")

    def load(self, file_path):
        data = torch.load(file_path)
        self.state_buffer = data['state_buffer'].to(self.device)
        self.knowledge_buffer = data['knowledge_buffer'].to(self.device)
        self.buffer_size = data['buffer_size']
        self.buffer_count = data['buffer_count']
        self.device = data['device']
        self.rng.bit_generator.state = data['rng_state']
        print(f"Buffer loaded from {file_path}")




class TrainableSyntheticBuffer(RolloutBuffer):
    def __init__(self, buffer_size=math.inf, synthetic_buffer_size=100, synthetic_init_threshold_size=1000, rng=None):
        super().__init__(buffer_size, rng)
        self.synthetic_buffer = []
        self.synthetic_buffer_size = synthetic_buffer_size
        self.synthetic_init_threshold_size = synthetic_init_threshold_size

    def init_synthetic_buffer(self):
        sample_idx = self.rng.choice(range(len(self.buffer)), size=self.synthetic_buffer_size, replace=False)
        sample_idx = np.array(sample_idx, dtype=int)
        self.synthetic_buffer = [self.buffer[i] for i in sample_idx]

    # This is outside of GradientMatchingBuffer class
    def update_synthetic_buffer(self):
        # Check Data condensation
        pass

    def get_synthetic_buffer(self):
        """
        Call this before sampling to get synthetic buffer and update the buffer.
        """
        return self.synthetic_buffer
    
    def sample(self, batch_size=1):
        """
        NOTE: Always update synthetic data before sample
        """
        if len(self.buffer) < self.synthetic_init_threshold_size:
            return []
        else:
            # First udpate synthetic buffer
            #self.update_synthetic_buffer()
            sample_idx = self.rng.choice(range(len(self.synthetic_buffer)), size=batch_size, replace=False)
            return [self.synthetic_buffer[i] for i in sample_idx]
    
    def add(self, observation):
        if len(self.buffer) == self.synthetic_init_threshold_size:
            self.init_synthetic_buffer()
        elif len(self.buffer) < self.synthetic_init_threshold_size:
            super().add(observation) # call the Rollout Buffer add
        else:
            pass



class TrainableSyntheticBufferBatch(RolloutBufferBatch):
    def __init__(self, state_dim, knowledge_dim=None, buffer_size=5000, synthetic_buffer_size=100, synthetic_init_threshold_size=1000, rng=None, device='cpu'):
        super().__init__(state_dim, knowledge_dim, buffer_size, device, rng)
        self.synthetic_buffer = torch.empty((synthetic_buffer_size, *state_dim), dtype=torch.float32, device=device)
        if knowledge_dim is None:
            self.synthetic_knowledge_buffer = torch.empty((synthetic_buffer_size, 1), dtype=torch.float32, device=device)
        else: 
            self.synthetic_knowledge_buffer = torch.empty((synthetic_buffer_size, *knowledge_dim), dtype=torch.float32, device=device)

        self.synthetic_buffer_size = synthetic_buffer_size
        self.synthetic_init_threshold_size = synthetic_init_threshold_size
        self.synthetic_buffer_count = 0
        self.is_synthetic_initialized = False
        self.is_init_optimizer = True
        self.device = device

    def init_synthetic_buffer(self):
        sample_idx = self.rng.choice(range(min(self.buffer_count, self.buffer_size)), size=self.synthetic_buffer_size, replace=False)
        #sample_idx = np.array(sample_idx, dtype=int)
        self.synthetic_buffer[:] = self.state_buffer[sample_idx]
        self.synthetic_knowledge_buffer[:] = self.knowledge_buffer[sample_idx]
        self.synthetic_buffer_count = len(sample_idx)
        self.is_synthetic_initialized = True

    # This is outside of GradientMatchingBuffer class
    def update_synthetic_buffer(self):
        # Check Data condensation
        pass

    def get_synthetic_buffer(self):
        """
        Call this before sampling to get synthetic buffer and update the buffer.
        """
        return self.synthetic_buffer
    
    def sample(self, batch_size=1):
        """
        NOTE: Always update synthetic data before sample
        """
        if self.buffer_count < self.synthetic_init_threshold_size:
            return [], []
        else:
            # First udpate synthetic buffer
            #self.update_synthetic_buffer()
            #sample_idx = self.rng.choice(range(len(self.synthetic_buffer)), size=batch_size, replace=False)
            #return [self.synthetic_buffer[i] for i in sample_idx]

            sample_indices = self.rng.choice(range(self.synthetic_buffer_count), size=batch_size, replace=False)

            return (self.synthetic_buffer[sample_indices], self.synthetic_knowledge_buffer[sample_indices])
    
    def add(self, observation):
        if self.buffer_count == self.synthetic_init_threshold_size:
            self.init_synthetic_buffer()
        elif self.buffer_count < self.synthetic_init_threshold_size:
            super().add(observation) # call the Rollout Buffer add
        else:
            pass

    def save(self, file_path):
        data = {
            'state_buffer': self.state_buffer,
            'knowledge_buffer': self.knowledge_buffer,
            'synthetic_buffer': self.synthetic_buffer,
            'synthetic_knowledge_buffer': self.synthetic_knowledge_buffer,
            'buffer_size': self.buffer_size,
            'synthetic_buffer_size': self.synthetic_buffer_size,
            'buffer_count': self.buffer_count,
            'synthetic_buffer_count': self.synthetic_buffer_count,
            'device': self.device,
            'rng_state': self.rng.bit_generator.state,
            'synthetic_init_threshold_size': self.synthetic_init_threshold_size,
            'is_synthetic_initialized': self.is_synthetic_initialized
        }
        torch.save(data, file_path)
        print(f"Buffer saved to {file_path}")

    def load(self, file_path):
        data = torch.load(file_path)
        self.state_buffer = data['state_buffer'].to(self.device)
        self.knowledge_buffer = data['knowledge_buffer'].to(self.device)
        self.synthetic_buffer = data['synthetic_buffer'].to(self.device)
        self.synthetic_knowledge_buffer = data['synthetic_knowledge_buffer'].to(self.device)
        self.buffer_size = data['buffer_size']
        self.synthetic_buffer_size = data['synthetic_buffer_size']
        self.buffer_count = data['buffer_count']
        self.synthetic_buffer_count = data['synthetic_buffer_count']
        self.synthetic_init_threshold_size = data['synthetic_init_threshold_size']
        self.is_synthetic_initialized = data['is_synthetic_initialized']
        self.rng.bit_generator.state = data['rng_state']
        print(f"Buffer loaded from {file_path}")
 
 
import random

from tree import SumTree

# The following PrioritizedReplayBuffer is adapted from https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
# NOTE: Only use it with DQN, which is run_student_per.py in the "src" folder.
class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, eps=1e-2, alpha=0.1, beta=0.1, device='cpu'):
        self.tree = SumTree(size=buffer_size)
        self.device = device

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.state = torch.empty((buffer_size, *state_size), dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty((buffer_size, *state_size), dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


# The following ReplayBuffer is adapted from https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
# NOTE: Only use it with DQN, which is run_student_per.py in the "src" folder.
class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, device):
        # state, action, reward, next_state, done
        self.state = torch.empty((buffer_size, *state_size), dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty((buffer_size, *state_size), dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch 
 

 
 
 
 

 
 
