# import os

from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.visualization import ModelVisualizer
from pydrake.all import (
    Simulator,
    StartMeshcat,
)

from manipulation import running_as_notebook
from manipulation.station import MakeHardwareStation, load_scenario

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

def create_scene():
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
        scenario_data += f"""
directives:
- add_model:
    name: iiwa
    file: package://manipulation/planar_iiwa14_no_collision.urdf
    default_joint_positions:
        iiwa_joint_2: [0.1]
        iiwa_joint_4: [-1.2]
        iiwa_joint_6: [1.6]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
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
- add_model:
    name: foam_brick
    file: package://manipulation/hydro/061_foam_brick.sdf
- add_model:
    name: robot_table
    file: package://manipulation/hydro/extra_heavy_duty_table_surface_only_collision.sdf
- add_weld:
    parent: world
    child: robot_table::link
    X_PC:
        translation: [0, 0, -0.7645]
- add_model:
    name: work_table
    file: package://manipulation/hydro/extra_heavy_duty_table_surface_only_collision.sdf
- add_weld:
    parent: world
    child: work_table::link
    X_PC:
        translation: [0.75, 0, -0.7645]
# Restrict the brick to move only in the x-z plane
- add_model:
    name: planar_joint
    file: package://manipulation/planar_joint.sdf
    default_joint_positions:
        planar_joint: [0.6, 0, 0]
- add_weld:
    parent: world
    child: planar_joint::parent
    X_PC:
        rotation: !Rpy {{ deg: [90, 0, 0]}}
- add_weld:
    parent: planar_joint::child
    child: foam_brick::base_link
model_drivers:
    iiwa: !IiwaDriver
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""

    scenario = load_scenario(data=scenario_data)
    station = MakeHardwareStation(scenario, meshcat)

    simulator = Simulator(station)
    meshcat.StartRecording()
    simulator.AdvanceTo(2.0)
    meshcat.PublishRecording()


create_scene()

while True:
    pass
