import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from typing import Unpack

from controllers.zone_managers.hrl.dqn import DQNNetwork
from controllers.zone_managers.hrl.memory import ReplayBuffer
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task
from models.zone import Zone
from utils.enums import Layer


class HRLZoneManager(ZoneManagerABC):
    def __init__(self, zone: Zone, dqn_lr=1e-3, sac_lr=3e-4, memory_size=10000):
        super().__init__(zone)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DQN (Low Level)
        self.state_dim_local = 6  # Task requirements + creator resources
        self.dqn = DQNNetwork(self.state_dim_local).to(self.device)
        self.dqn_target = DQNNetwork(self.state_dim_local).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=dqn_lr)
        self.dqn_memory = ReplayBuffer(memory_size)
        self.dqn_target_update = 10
        self.dqn_update_counter = 0

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64

        # Episode tracking
        self.dqn_current_state = None
        self.dqn_current_action = None
        self.all_possible_nodes = {}

    def _get_local_state(self, task: Task) -> torch.Tensor:
        """Get state representation for local offloading decision."""
        state = [
            # Task parameters
            task.power / 100.0,
            task.exec_time / 100.0,
            task.deadline / 1000.0,
            # Creator parameters
            task.creator.remaining_power / 100.0,
            len(task.creator.tasks),
            task.creator.max_tasks_queue_len,
        ]
        return torch.FloatTensor(state).to(self.device)

    def _select_action(self, state: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            q_values = self.dqn(state.unsqueeze(0))
            return q_values.argmax().item()

    def _update_network(self):
        if len(self.dqn_memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.dqn_memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.float().to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.float().to(self.device)

        # Compute current Q values
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.dqn_target(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        # Update target network
        self.dqn_update_counter += 1
        if self.dqn_update_counter % self.dqn_target_update == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def can_offload_task(self, task: Task) -> bool:
        # Update high level heuristic
        all_fog_nodes = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        self.all_possible_nodes = {
            node_id: node for node_id, node in all_fog_nodes.items()
            if node.can_offload_task(task)
        }

        # Update lower level RL state
        self.dqn_current_state = self._get_local_state(task)
        self.dqn_current_action = self._select_action(self.dqn_current_state)
        if self.dqn_current_action == 0 and task.creator.can_offload_task(task):
            return True
        return len(self.all_possible_nodes) > 0

    def assign_task(self, task: Task) -> FogLayerABC:
        # Action 0: Process locally if possible
        if self.dqn_current_action == 0 and task.creator.can_offload_task(task):
            return task.creator

        # Action 1 or can't process locally: Select nearest fog node
        nearest_node = None
        min_distance = float('inf')

        for node in self.all_possible_nodes.values():
            distance = self._calculate_distance(task.creator, node)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    @staticmethod
    def _calculate_distance(creator, node) -> float:
        return np.sqrt((creator.x - node.x) ** 2 + (creator.y - node.y) ** 2)

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        if self.dqn_current_state is None:
            return

        current_task = kwargs.get('current_task')
        if not current_task:
            return

        # Calculate reward
        reward = self._calculate_reward(current_task)

        # Get next state
        next_state = self._get_local_state(current_task)

        # Store transition
        done = current_task.is_completed or current_task.is_deadline_missed
        self.dqn_memory.push(
            self.dqn_current_state,
            self.dqn_current_action,
            reward,
            next_state,
            done
        )

        # Update network
        self._update_network()

        if done:
            self.dqn_current_state = None
            self.dqn_current_action = None

    @staticmethod
    def _calculate_reward(task: Task) -> float:
        reward = 0.0

        # Small negative reward per time step to encourage faster completion
        reward -= 0.1

        # Check task status
        if task.executor == task.creator:
            # Ongoing local processing
            reward += 0.5  # Small positive reward to consider local processing
            cpu_util = 100.0 - task.creator.remaining_power / task.power * 100.0
            if cpu_util > 90:
                reward -= 1.0  # Penalty for high resource utilization

        # Completed or failed conditions
        if task.has_migrated:
            reward -= 5.0  # Penalty for task migration

            if task.executor.layer == Layer.CLOUD:
                reward -= 5.0  # Penalty for offloading to cloud

        if task.is_completed:
            completion_time = task.finish_time
            deadline_margin = task.deadline - completion_time
            reward += 20.0 + max(0.0, deadline_margin / task.deadline) * 10.0

            if task.executor == task.creator:
                reward += 10.0  # Bonus for successful local completion

        if task.is_deadline_missed:
            reward -= 50.0

        return reward
