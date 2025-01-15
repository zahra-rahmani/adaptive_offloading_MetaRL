from collections import defaultdict
from typing import Dict, List

from config import Config
from controllers.loader import Loader
from controllers.metric import MetricsController
from controllers.zone_managers.base import ZoneManagerABC
from models.node.base import MobileNodeABC, NodeABC
from models.node.cloud import CloudNode
from models.node.fog import FixedFogNode
from models.node.fog import MobileFogNode
from models.node.user import UserNode
from models.task import Task
from utils.clock import Clock
from utils.enums import Layer


class Simulator:
    def __init__(self, loader: Loader, clock: Clock, cloud: CloudNode):
        self.metrics: MetricsController = MetricsController()
        self.loader: Loader = loader
        self.cloud_node: CloudNode = cloud
        self.zone_managers: Dict[str, ZoneManagerABC] = {}
        self.fixed_fog_nodes: Dict[str, FixedFogNode] = {}
        self.mobile_fog_nodes: Dict[str, MobileFogNode] = {}
        self.user_nodes: Dict[str, UserNode] = {}
        self.clock: Clock = clock
        self.task_zone_managers: Dict[str, ZoneManagerABC] = {}

    def init_simulation(self):
        self.clock.set_current_time(0.0)
        self.zone_managers = self.loader.load_zones()
        self.fixed_fog_nodes = self.loader.load_fixed_zones()
        self.assign_fixed_nodes()
        self.update_mobile_fog_nodes_coordinate()
        self.update_user_nodes_coordinate()

    def assign_fixed_nodes(self):
        for z_id, zone_manager in self.zone_managers.items():
            fixed_nodes: List[FixedFogNode] = []
            for n_id, fixed_node in self.fixed_fog_nodes.items():
                if zone_manager.zone.is_in_coverage(fixed_node.x, fixed_node.y):
                    fixed_nodes.append(fixed_node)
            zone_manager.add_fixed_fog_nodes(fixed_nodes)

    def start_simulation(self):
        self.init_simulation()
        while (current_time := self.clock.get_current_time()) < Config.SimulatorConfig.SIMULATION_DURATION:
            nodes_tasks = self.load_tasks(current_time)
            user_possible_zones = self.assign_mobile_nodes_to_zones(self.user_nodes, layer=Layer.USER)
            mobile_possible_zones = self.assign_mobile_nodes_to_zones(self.mobile_fog_nodes, layer=Layer.FOG)

            merged_possible_zones: Dict[str, List[ZoneManagerABC]] = {**user_possible_zones, **mobile_possible_zones}
            for creator_id, tasks in nodes_tasks.items():
                zone_managers = merged_possible_zones[creator_id]
                for task in tasks:
                    self.metrics.inc_total_tasks()
                    has_offloaded = False
                    for zone_manager in zone_managers:
                        if zone_manager.can_offload_task(task):
                            has_offloaded = True
                            assignee = zone_manager.offload_task(task, current_time)

                            self.metrics.inc_node_tasks(assignee.id)
                            break
                    if not has_offloaded:
                        self.offload_to_cloud(task, current_time)

            self.update_graph()
            self.execute_tasks_for_one_step()
            self.metrics.flush()

            self.metrics.log_metrics()
        self.drop_not_completed_tasks()

    def load_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        tasks: Dict[str, List[Task]] = defaultdict(list)
        for creator_id, creator_tasks in self.loader.load_nodes_tasks(current_time).items():
            creator = None
            if creator_id in self.user_nodes:
                creator = self.user_nodes[creator_id]
            elif creator_id in self.mobile_fog_nodes:
                creator = self.mobile_fog_nodes[creator_id]
            assert creator is not None

            for task in creator_tasks:
                task.creator = creator
                tasks[creator_id].append(task)
        return tasks

    def execute_tasks_for_one_step(self):
        executed_tasks: List[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            self.cloud_node.id: self.cloud_node,
        }

        for node_id, node in merged_nodes.items():
            tasks = node.execute_tasks(self.clock.get_current_time())
            executed_tasks.extend(tasks)
            for task in tasks:
                zone_manager = self.task_zone_managers.get(task.id)
                if zone_manager:
                    zone_manager.update(current_task=task)

                if task.has_migrated:
                    self.metrics.inc_migration()
                if task.is_deadline_missed:
                    self.metrics.inc_deadline_miss()
                else:
                    self.metrics.inc_completed_task()

    def update_graph(self):
        self.clock.tick()
        self.update_user_nodes_coordinate()
        self.update_mobile_fog_nodes_coordinate()

    def offload_to_cloud(self, task: Task, current_time: float):
        if self.cloud_node.can_offload_task(task):
            self.cloud_node.offload_task(task, current_time)
            self.metrics.inc_cloud_tasks()
        else:
            self.metrics.inc_deadline_miss()

    def assign_mobile_nodes_to_zones(
            self,
            mobile_nodes: dict[str, MobileNodeABC],
            layer: Layer
    ) -> Dict[str, List[ZoneManagerABC]]:

        nodes_possible_zones: Dict[str, List[ZoneManagerABC]] = defaultdict(list)
        for z_id, zone_manager in self.zone_managers.items():
            nodes: List[MobileNodeABC] = []
            for n_id, mobile_node in mobile_nodes.items():
                if zone_manager.zone.is_in_coverage(mobile_node.x, mobile_node.y):
                    nodes.append(mobile_node)
                    nodes_possible_zones[n_id].append(zone_manager)
            if layer == Layer.FOG:
                zone_manager.set_mobile_fog_nodes(nodes)
        return nodes_possible_zones

    def update_mobile_fog_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_mobile_fog_nodes(self.clock.get_current_time())
        self.mobile_fog_nodes = self.update_nodes_coordinate(self.mobile_fog_nodes, new_nodes_data)

    def update_user_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_user_nodes(self.clock.get_current_time())
        self.user_nodes = self.update_nodes_coordinate(self.user_nodes, new_nodes_data)

    @staticmethod
    def update_nodes_coordinate(old_nodes: dict[str, MobileNodeABC], new_nodes: dict[str, MobileNodeABC]):
        data: Dict[str, MobileNodeABC] = {}
        for n_id, new_node in new_nodes.items():
            if n_id not in old_nodes:
                node = new_node
            else:
                node = old_nodes[n_id]
                node.x = new_node.x
                node.y = new_node.y
                node.angle = new_node.angle
                node.speed = new_node.speed
            data[n_id] = node
        return data

    def drop_not_completed_tasks(self) -> List[Task]:
        left_tasks: list[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            self.cloud_node.id: self.cloud_node,
        }

        for node_id, node in merged_nodes.items():
            left_tasks.extend(node.tasks)
            for i in range(len(node.tasks)):
                self.metrics.inc_deadline_miss()
        return left_tasks
