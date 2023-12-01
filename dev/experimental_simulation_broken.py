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
 
your_model_filename = full_path + "Pound_Cake_OBJ.sdf"

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
"""

def add_random_obj(ranges, name, cnt):
    global scenario_data
    for num in range(cnt):
        scenario_data += f"""
- add_model:
    name: {name}_{num}
    file: file://{full_path}{name}.sdf
    default_free_body_pose:
        base_link:
            translation: [{ranges['x'] + np.random.randint(-10, 10)/50}, {ranges['y'] + np.random.randint(-10, 10)/50}, {ranges['z'] + np.random.randint(10)/10}]

""" 

def add_custom_random_obj(ranges, name, cnt):
    global scenario_data
    for num in range(cnt):
        scenario_data += f"""
- add_model:
    name: {name}_{num}
    file: file://{full_path}{name}.sdf
    default_free_body_pose:
        {name}: # Change here!
            translation: [{0.5 + np.random.randint(-10, 10)/50}, {-0.6 + np.random.randint(-10, 10)/50}, {0.01}] 
            rotation: !Rpy {{ deg: [{np.random.randint(0, 90)}, {np.random.randint(0, 90)}, {np.random.randint(0, 90)}] }}
"""
   
add_random_obj({"x": 0, "y": -0.6, "z": 0.2}, "foam_chicken", 10)
# add_custom_random_obj({"x": 0.2, "y": -0.6, "z": 0.2}, "Pound_Cake_OBJ", 2)

scenario_data += f"""
model_drivers:
    iiwa: !IiwaDriver
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""

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

input("Simulation Running at http://localhost:7000 — Press Any Key to Terminate ")
