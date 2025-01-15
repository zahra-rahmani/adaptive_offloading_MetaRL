from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from config import Config
from models.base import ModelBaseABC
from utils.enums import Layer


@dataclass
class NodeABC(ModelBaseABC, abc.ABC):
    """
    Represents any computational resource in the system. Nodes can belong to different layers,
    such as User, Fog, or Cloud.
    """

    x: float = 0
    y: float = 0
    power: float = 0  # The computational power available at this node.
    radius: float = 0  # The radius that this node can cover.

    remaining_power: float = 0  # The amount of computational resourced left after executing current tasks.
    tasks: Deque = field(default_factory=deque)  # The list of tasks that are currently offloaded in this node.

    def can_offload_task(self, task) -> bool:
        """Checks whether the task can be offloaded in this node."""
        if len(self.tasks) >= self.max_tasks_queue_len:
            return False
        if task.power > self.remaining_power:
            return False
        return True

    def offload_task(self, task, current_time: float):
        """Offload a task in the current node."""
        if not self.can_offload_task(task):
            raise Exception(
                f"Task cannot be offloaded: Node ID={self.id}, "
                f"Remaining Power={self.remaining_power}, Task Power={task.power}, "
                f"Queue Length={len(self.tasks)}/{self.max_tasks_queue_len}"
            )

        self.tasks.append(task)
        self.remaining_power -= task.power
        task.start_time = current_time
        task.executor = self

    def execute_tasks(self, current_time: float) -> list:
        """Execute current tasks and return all completed tasks."""
        # TODO: This method is executed per second, may be in further enhancements it is better use thread to execute
        #  tasks.
        executed_tasks = []
        remaining_tasks = deque()
        while self.tasks:
            task = self.tasks.popleft()
            if current_time - task.start_time >= task.real_exec_time:
                task.finish_time = task.start_time + task.real_exec_time
                self.remaining_power += task.power
                executed_tasks.append(task)
            else:
                remaining_tasks.append(task)

        self.tasks = remaining_tasks
        return executed_tasks

    @property
    @abc.abstractmethod
    def layer(self) -> Layer:
        """The layer to which this node belongs (User, Fog, or Cloud)."""
        raise NotImplemented

    @property
    @abc.abstractmethod
    def max_tasks_queue_len(self) -> int:
        """The max number of tasks that this node can process in parallel."""
        raise NotImplementedError


@dataclass
class MobileNodeABC(NodeABC, abc.ABC):
    """
    Represents any computational resource in the system which can move from one location to another. Mobile nodes may
    belong to different layers, such as User or Fog.
    """

    speed: float = 0
    angle: float = 0
