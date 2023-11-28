from pydrake.all import ModelVisualizer, StartMeshcat, PackageMap, Simulator

from manipulation import running_as_notebook, ConfigureParser
from manipulation.station import load_scenario, MakeHardwareStation

# Start the visualizer.
meshcat = StartMeshcat()

visualizer = ModelVisualizer(meshcat=meshcat)
ConfigureParser(visualizer.parser())
visualizer.AddModels(
    url="package://manipulation/mobile_iiwa14_primitive_collision.urdf"
)
visualizer.Run(loop_once=not running_as_notebook)
meshcat.DeleteAddedControls()

scenario_data = """
directives:
- add_model:
    name: mobile_iiwa
    file: package://manipulation/mobile_iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
model_drivers:
    mobile_iiwa+wsg: !InverseDynamicsDriver {}
"""

scenario = load_scenario(data=scenario_data)
station = MakeHardwareStation(scenario, meshcat)
simulator = Simulator(station)
context = simulator.get_mutable_context()
x0 = station.GetOutputPort("mobile_iiwa.state_estimated").Eval(context)
station.GetInputPort("mobile_iiwa.desired_state").FixValue(context, x0)
simulator.AdvanceTo(0.1)

input("testing testing")
