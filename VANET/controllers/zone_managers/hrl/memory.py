import torch
import random

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(states), torch.tensor(actions),
                torch.tensor(rewards), torch.stack(next_states),
                torch.tensor(dones))

    def __len__(self):
        return len(self.buffer)
