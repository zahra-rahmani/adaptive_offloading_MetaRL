from models.node.cloud import CloudNode
from controllers.loader import Loader
from controllers.simulator import Simulator
from config import Config
from utils.clock import Clock


loader = Loader(
    zone_file="./data/hamburg.zon.xml",
    fixed_fn_file="./data/hamburg.fn.xml",
    mobile_file="./data/hamburg.out.xml",
    task_file="./data/hamburg.tasks.xml"
)

cloud = CloudNode(
    id="CLOUD0",
    x=Config.CloudConfig.DEFAULT_X,
    y=Config.CloudConfig.DEFAULT_Y,
    power=Config.CloudConfig.DEFAULT_COMPUTATION_POWER,
    remaining_power=Config.CloudConfig.DEFAULT_COMPUTATION_POWER,
    radius=Config.CloudConfig.DEFAULT_RADIUS,
)

simulator = Simulator(loader, Clock(), cloud)

simulator.start_simulation()
