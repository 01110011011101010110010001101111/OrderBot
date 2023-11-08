"""
script that pulls up an iiwa
"""

import numpy as np
import pydot
from IPython.display import HTML, SVG, display
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    GenerateHtml,
    InverseDynamicsController,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    Simulator,
    StartMeshcat,
)

from manipulation import running_as_notebook

running_as_notebook = True

meshcat = StartMeshcat()

plant = MultibodyPlant(time_step=1e-4)
Parser(plant).AddModelsFromUrl(
    "package://drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"
)
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
plant.Finalize()

context = plant.CreateDefaultContext()
print(context)

# Set all of the joint positions at once in a single vector.
plant.SetPositions(context, [-1.57, 0.1, 0, 0, 0, 1.6, 0])
# You can also set them by referencing particular joints.
plant.GetJointByName("iiwa_joint_4").set_angle(context, -1.2)

plant.get_actuation_input_port().FixValue(context, np.zeros(7))

simulator = Simulator(plant, context)
simulator.AdvanceTo(5.0)

meshcat.Delete()
meshcat.DeleteAddedControls()
builder = DiagramBuilder()

# Adds both MultibodyPlant and the SceneGraph, and wires them together.
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
# Note that we parse into both the plant and the scene_graph here.
Parser(plant, scene_graph).AddModelsFromUrl(
    "package://drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"
)
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
plant.Finalize()

# Adds the MeshcatVisualizer and wires it to the SceneGraph.
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

diagram = builder.Build()
diagram.set_name("plant and scene_graph")

context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)

plant_context = plant.GetMyMutableContextFromRoot(context)
plant.SetPositions(plant_context, [-1.57, 0.1, 0, -1.2, 0, 1.6, 0])
plant.get_actuation_input_port().FixValue(plant_context, np.zeros(7))

simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(5.0 if running_as_notebook else 0.1)



input("Press any key to stop. ")
