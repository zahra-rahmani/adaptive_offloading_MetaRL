class Config:
    SIMULATION_DURATION = 10

    class CloudConfig:
        DEFAULT_X = 1000
        DEFAULT_Y = 1000
        DEFAULT_RADIUS = 1000

        MAX_TASK_QUEUE_LEN = 2000
        DEFAULT_COMPUTATION_POWER = 50


    class FixedFogNodeConfig:
        MAX_TASK_QUEUE_LEN = 200
        DEFAULT_COMPUTATION_POWER = 20

    class MobileFogNodeConfig:
        DEFAULT_RADIUS = 10
        MAX_TASK_QUEUE_LEN = 100
        DEFAULT_COMPUTATION_POWER = 10

    class UserFogNodeConfig:
        MAX_TASK_QUEUE_LEN = 5
        DEFAULT_COMPUTATION_POWER = 5

    class SimulatorConfig:
        SIMULATION_DURATION = 1199

    class ZoneManagerConfig:
        MAX_TASK_BUFFER = 100

        ALGORITHM_RANDOM = "random"
        ALGORITHM_HEURISTIC = "heuristic"
        ALGORITHM_HRL = "hrl"
        ALGORITHM_META_RL = "meta_rl"

        DEFAULT_ALGORITHM = ALGORITHM_META_RL

    class RandomZoneManagerConfig:

        OFFLOAD_CHANCE: float = 0.5

    class TaskConfig:
        PACKET_COST_PER_METER = 0.001
        TASK_COST_PER_METER = 0.005
        MIGRATION_OVERHEAD = 0.01
        CLOUD_PROCESSING_OVERHEAD = 0.5
