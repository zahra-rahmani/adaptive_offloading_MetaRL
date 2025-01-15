from dataclasses import dataclass

import numpy as np

from models.base import ModelBaseABC
from models.node.base import NodeABC
from utils.enums import Layer
from config import Config


@dataclass
class Task(ModelBaseABC):
    """Represents tasks that each mobile node can produce over time."""
    release_time: float
    deadline: float
    exec_time: float  # The amount of time that this task required to execute.
    power: float  # The amount of power unit that this tasks consumes while executing.
    creator_id: str  # Thd id of the node who created the task.

    start_time: float = 0  # The time that this task was offloaded to a node (either local or external).
    finish_time: float = 0  # The time that this task was finished in the offloaded node.

    creator: NodeABC = None
    executor: NodeABC = None

    @property
    def real_exec_time(self) -> float:
        is_cloud = self.executor.layer == Layer.CLOUD
        distance = self.get_creator_and_executor_distance()
        has_migrated = self.has_migrated

        real_exec_time = self.exec_time + \
                            distance * 2 * Config.TaskConfig.PACKET_COST_PER_METER + \
                            distance * Config.TaskConfig.TASK_COST_PER_METER
        if is_cloud:
            real_exec_time = Config.TaskConfig.CLOUD_PROCESSING_OVERHEAD
        elif has_migrated:
            real_exec_time += Config.TaskConfig.MIGRATION_OVERHEAD * distance

        return real_exec_time

    @property
    def has_migrated(self) -> bool:
        if self.creator.id == self.executor.id:
            return False
        elif self.executor.radius > self.get_creator_and_executor_distance():
            return False
        return True

    def get_creator_and_executor_distance(self) -> float:
        return np.sqrt((self.creator.x - self.executor.x) ** 2 + (self.creator.y - self.executor.y) ** 2)

    @property
    def is_deadline_missed(self) -> bool:
        return self.deadline < self.finish_time

    @property
    def is_completed(self) -> bool:
        return self.finish_time > 0
