from collections import defaultdict
from typing import Type

from controllers.zone_managers.base import ZoneManagerABC
from controllers.zone_managers.heuristic import HeuristicZoneManager
from controllers.zone_managers.random import RandomZoneManager
from controllers.zone_managers.hrl import HRLZoneManager
from utils.xml_parser import *

from controllers.zone_managers.meta_rl import MetaRLZoneManager



class Loader:
    # TODO: Fill this map after adding zone managers completed.
    ALGORITHM_MAP: Dict[str, Type[ZoneManagerABC]] = {
        Config.ZoneManagerConfig.ALGORITHM_RANDOM: RandomZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_HEURISTIC: HeuristicZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_HRL: HRLZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_META_RL: MetaRLZoneManager,  # Add Meta-RL
    }

    def __init__(self, zone_file: str, fixed_fn_file: str, mobile_file: str, task_file: str):
        self.zone_parser = ZoneSumoXMLParser(zone_file)
        self.fixed_fn_parser = FixedFogNodeSumoXMLParser(fixed_fn_file)
        self.mobile_node_parser = MobileNodeSumoXMLParser(mobile_file)
        self.task_parser = TaskSumoXMLParser(task_file)

    def load_zones(self) -> Dict[str, ZoneManagerABC]:
        zone_managers: Dict[str, ZoneManagerABC] = {}

        zones = self.zone_parser.parse()
        for zone in zones:
            zone_manager = self.ALGORITHM_MAP[Config.ZoneManagerConfig.DEFAULT_ALGORITHM]
            zone_managers[zone.id] = zone_manager(zone)
        return zone_managers

    def load_fixed_zones(self) -> Dict[str, FixedFogNode]:
        fixed_fog_nodes: Dict[str, FixedFogNode] = {}

        for fixed_node in self.fixed_fn_parser.parse():
            fixed_fog_nodes[fixed_node.id] = fixed_node
        return fixed_fog_nodes

    def load_mobile_fog_nodes(self, time_step: float) -> Dict[str, MobileFogNode]:
        mobile_fog_nodes: Dict[str, MobileFogNode] = {}

        for mobile_node in self.mobile_node_parser.parse()[time_step][1]:
            mobile_fog_nodes[mobile_node.id] = mobile_node
        return mobile_fog_nodes

    def load_user_nodes(self, time_step: float) -> Dict[str, UserNode]:
        user_fog_nodes: Dict[str, UserNode] = {}

        for user_node in self.mobile_node_parser.parse()[time_step][0]:
            user_fog_nodes[user_node.id] = user_node
        return user_fog_nodes

    def load_nodes_tasks(self, time_step: float) -> Dict[str, List[Task]]:
        tasks: Dict[str, List[Task]] = defaultdict(list)

        for task in self.task_parser.parse().get(time_step, []):
            tasks[task.creator_id].append(task)
        return tasks
