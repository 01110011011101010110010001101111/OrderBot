import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
    DoDifferentialInverseKinematics,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.visualization import MeshcatPoseSliders

from manipulation import running_as_notebook
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK, ExtractBodyPose
from manipulation.station import MakeHardwareStation, load_scenario

from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.visualization import ModelVisualizer
from pydrake.all import (
    Simulator,
    StartMeshcat,
)

from manipulation import running_as_notebook
from manipulation.station import MakeHardwareStation, load_scenario



# Start the visualizer.
meshcat = StartMeshcat()

full_path = "/Users/paromitadatta/Desktop/64210/6.4210-Final-Project/objects/"

table_top_sdf_file = full_path + "table_top.sdf"
table_top_sdf = """<?xml version="1.0"?>
<sdf version="1.7">

  <model name="table_top">
    <link name="table_top_link">
      <inertial>
        <mass>18.70</mass>
        <inertia>
          <ixx>0.79</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.53</iyy>
          <iyz>0</iyz>
          <izz>1.2</izz>
        </inertia>
      </inertial>
    <visual name="bottom">
        <pose>0.0 0.0 0.445 0 0 0</pose>
        <geometry>
          <box>
            <size>1.49 1.63 0.015</size>
          </box>
        </geometry>
        <material>
          <diffuse>0.9 0.9 0.9 1.0</diffuse>
        </material>
      </visual>
      <collision name="bottom">
        <pose>0.0 0.0 0.445 0 0 0</pose>
        <geometry>
          <box>
            <size>0.49 0.63 0.015</size>
          </box>
        </geometry>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>1.0e6</drake:hydroelastic_modulus>
        </drake:proximity_properties>
      </collision>
    </link>
    <frame name="table_top_center">
      <pose relative_to="table_top_link">0 0 0.47 0 0 0</pose>
    </frame>
  </model>
</sdf>

"""

with open(table_top_sdf_file, "w") as f:
    f.write(table_top_sdf)


your_model_filename = full_path + "Pound_Cake_OBJ.sdf"

if your_model_filename:
    visualizer = ModelVisualizer(meshcat=meshcat)
    visualizer.parser().AddModels(your_model_filename)
    visualizer.Run(loop_once=True)

scenario_data = f"""
directives:
- add_model:
    name: table_top
    file: file://{table_top_sdf_file}
- add_weld:
    parent: world
    child: table_top::table_top_center
"""
if your_model_filename:
    scenario_data += "directives:"
    scenario_data += f"""

- add_model:
    name: table_top
    file: file://{table_top_sdf_file}
- add_weld:
    parent: world
    child: table_top::table_top_center
- add_model:
    name: your_model
    file: file://{your_model_filename}
    default_free_body_pose:
        Pound_Cake_OBJ: # Change here!
            translation: [0, 0, 0.5]
            rotation: !Rpy {{ deg: [42, 33, 18] }}
"""

scenario = load_scenario(data=scenario_data)
station = MakeHardwareStation(scenario, meshcat)

simulator = Simulator(station)
meshcat.StartRecording()
simulator.AdvanceTo(2.0)
meshcat.PublishRecording()


scenario_data = f"""
directives:
- add_directives:
    file: package://manipulation/iiwa_and_wsg.dmd.yaml
- add_model:
    name: bin0
    file: file://{full_path}bin.sdf
- add_weld:
    parent: world
    child: bin0::bin_base
    X_PC:
      rotation: !Rpy {{ deg: [0.0, 0.0, 180.0 ]}}
      translation: [-0.05, -0.5, -0.015]
- add_model:
    name: bin1
    file: package://manipulation/hydro/bin.sdf
- add_weld:
    parent: world
    child: bin1::bin_base
    X_PC:
      rotation: !Rpy {{ deg: [0.0, 0.0, 180.0 ]}}
      translation: [0.5, -0.5, -0.015]
- add_model:
    name: bin2
    file: package://manipulation/hydro/bin.sdf
- add_weld:
    parent: world
    child: bin2::bin_base
    X_PC:
      rotation: !Rpy {{ deg: [0.0, 0.0, 180.0 ]}}
      translation: [0.5, 0.25, -0.015]

- add_model:
    name: floor
    file: package://manipulation/floor.sdf
- add_weld:
    parent: world
    child: floor::box
    X_PC:
        translation: [0, 0, -.5]
- add_model:
    name: foam_brick
    file: file://{full_path}foam_chicken.sdf
    default_free_body_pose:
        base_link:
            translation: [0, -0.6, 0.2]
- add_model:
    name: your_model
    file: file://{your_model_filename}
    default_free_body_pose:
        Pound_Cake_OBJ: # Change here!
            translation: [0, -0.6, 0.2]
            rotation: !Rpy {{ deg: [42, 33, 18] }}
model_drivers:
    iiwa: !IiwaDriver
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""

# def add_random_obj(ranges, name, cnt):
#     global scenario_data
#     for num in range(cnt):
#         scenario_data += f"""
# 
# directives:
# - add_model:
#     name: {name}_{num}
#     file: file://{full_path}{name}.sdf
#     default_free_body_pose:
#         base_link:
#             translation: [0, {-0.6 + np.random.randint(-10, 10)/10}, {0.2 + np.random.randint(-10, 10)/10}]
# 
# """ 
#    
# add_random_obj(None, "foam_chicken", 100)

print(scenario_data)

def teleop_3d():
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    # TODO(russt): Replace with station.AddDiffIk(...)
    controller_plant = station.GetSubsystemByName(
        "iiwa.controller"
    ).get_multibody_plant_for_control()
    # Set up differential inverse kinematics.
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName("iiwa_link_7"),
    )
    builder.Connect(
        differential_ik.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        differential_ik.GetInputPort("robot_state"),
    )

    # Set up teleop widgets.
    meshcat.DeleteAddedControls()
    teleop = builder.AddSystem(
        MeshcatPoseSliders(
            meshcat,
            lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
        )
    )
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    # Note: This is using "Cheat Ports". For it to work on hardware, we would
    # need to construct the initial pose from the HardwareStation outputs.
    plant = station.GetSubsystemByName("plant")
    ee_pose = builder.AddSystem(
        ExtractBodyPose(
            station.GetOutputPort("body_poses"),
            plant.GetBodyByName("iiwa_link_7").index(),
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"), ee_pose.get_input_port()
    )
    builder.Connect(ee_pose.get_output_port(), teleop.get_input_port())
    wsg_teleop = builder.AddSystem(WsgButton(meshcat))
    builder.Connect(
        wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.get_mutable_context()

    if True:  # Then we're not just running as a test on CI.
        simulator.set_target_realtime_rate(1.0)

        meshcat.AddButton("Stop Simulation", "Escape")
        print("Press Escape to stop the simulation")
        while meshcat.GetButtonClicks("Stop Simulation") < 1:
            simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        meshcat.DeleteButton("Stop Simulation")

    else:
        simulator.AdvanceTo(0.1)


teleop_3d()

input("ahhhhhhhhhh ")
