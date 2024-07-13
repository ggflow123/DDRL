from students import *
from unittest import TestCase
import gymnasium as gym

class TestPolicyStudent(TestCase):
    def setUp(self):
        self.env = gym.make("CartPole-v1")
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]

        self.student = PolicyStudent(state_size=self.state_size, action_size=self.action_size)

    def test_step(self):
        state = self.env.reset()
        state = torch.tensor(state[0], dtype=torch.float32)
        action = self.student.step(state)
        self.assertTrue(action in range(self.action_size))

    def test_get_knowledge(self):
        state = self.env.reset()
        state = torch.tensor(state[0], dtype=torch.float32)
        knowledge = self.student.get_knowledge(state)
        self.assertEqual(knowledge.shape, torch.Size([self.action_size]))


    