import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Unpack, List

from random import sample

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC, FixedFogNode, MobileFogNode
from models.task import Task
from models.zone import Zone
from utils.enums import Layer


class MetaRLZoneManager(ZoneManagerABC):
    def __init__(self, zone: Zone, meta_lr=1e-3, task_lr=1e-4, memory_capacity=10000):
        super().__init__(zone)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Meta-learning model
        self.meta_model = self._build_model().to(self.device)
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=meta_lr)

        # Task-specific model
        self.task_model = self._build_model().to(self.device)
        self.task_model.load_state_dict(self.meta_model.state_dict())
        self.task_optimizer = optim.Adam(self.task_model.parameters(), lr=task_lr)

        # Replay buffer
        self.memory = []
        self.memory_capacity = memory_capacity

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Episode tracking
        self.current_state = None
        self.current_action = None
        self.all_possible_nodes = {}

    def _build_model(self):
        """Define the Meta-RL model architecture."""
        return nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Actions: local (0) or offload (1)
        )

    def can_offload_task(self, task: Task) -> bool:
        """Determine whether the task can be offloaded."""
        self.current_state = self._get_state(task)
        self.current_action = self._select_action(self.current_state)

        if self.current_action == 0:  # Local processing
            return task.creator.can_offload_task(task)

        # Determine possible fog nodes for offloading
        all_fog_nodes = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        self.all_possible_nodes = {
            node_id: node for node_id, node in all_fog_nodes.items()
            if node.can_offload_task(task)
        }
        return len(self.all_possible_nodes) > 0

    def assign_task(self, task: Task) -> FogLayerABC:
        """Assign the task based on the selected action."""
        if self.current_action == 0:  # Local processing
            if task.creator.can_offload_task(task):
                return task.creator

        # Find nearest fog node
        nearest_node = None
        nearest_distance = float('inf')
        for node in self.all_possible_nodes.values():
            distance = self._calculate_distance(task.creator, node)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node = node

        return nearest_node or task.creator  # Default to local if no fog node is available

    def _get_state(self, task: Task) -> torch.Tensor:
        """Construct state representation from task and node features."""
        return torch.FloatTensor([
            task.power / 100.0,
            task.exec_time / 100.0,
            task.deadline / 1000.0,
            task.creator.remaining_power / 100.0,
            len(task.creator.tasks),
            task.creator.max_tasks_queue_len,
        ]).to(self.device)

    def _select_action(self, state: torch.Tensor) -> int:
        """Select an action based on the current policy."""
        if np.random.rand() < self.epsilon:  # Exploration
            return np.random.randint(0, 2)

        with torch.no_grad():  # Exploitation
            q_values = self.task_model(state.unsqueeze(0))
            return q_values.argmax().item()

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        """Update task-specific and meta-models."""
        current_task = kwargs.get('current_task')
        if not current_task:
            return

        reward = self._calculate_reward(current_task)
        next_state = self._get_state(current_task)
        done = current_task.is_completed or current_task.is_deadline_missed

        # Store experience
        self.memory.append((self.current_state, self.current_action, reward, next_state, done))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

        # Update task-specific model
        self._update_task_model()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update_task_model(self):
        """Update the task-specific model using experiences."""
        if len(self.memory) < 32:  # Batch size
            return

        # Sample batch
        batch = sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        # Compute Q-values and targets
        q_values = self.task_model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q = self.task_model(next_states).max(1)[0]
            targets = rewards + self.gamma * (1 - dones) * max_next_q

        # Compute loss and update
        loss = nn.functional.mse_loss(q_values.squeeze(), targets)
        self.task_optimizer.zero_grad()
        loss.backward()
        self.task_optimizer.step()

    @staticmethod
    def _calculate_distance(creator, node) -> float:
        """Calculate the distance between the task creator and a node."""
        return np.sqrt((creator.x - node.x) ** 2 + (creator.y - node.y) ** 2)

    @staticmethod
    def _calculate_reward(task: Task) -> float:
        """Define the reward function for task performance."""
        reward = -0.1  # Time penalty
        if task.is_completed:
            reward += 20.0
            reward += max(0, (task.deadline - task.finish_time) / task.deadline) * 10.0
        if task.is_deadline_missed:
            reward -= 50.0
        if task.has_migrated:
            reward -= 5.0
        return reward
