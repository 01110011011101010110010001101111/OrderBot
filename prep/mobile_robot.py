import numpy as np
from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    MeshcatVisualizer,
    StartMeshcat,
    MeshcatVisualizerParams,
    Simulator,
    JointSliders,
    InverseKinematics,
    RotationMatrix,
    Solve,
    ContactVisualizerParams,
    ContactVisualizer,
    GeometrySet,
    CollisionFilterDeclaration,
    Role,
    eq,
    RigidTransform,
)
from manipulation import ConfigureParser, running_as_notebook
import pydot
from IPython.display import display, Image, SVG

## override
running_as_notebook = True

# Start the visualizer.
meshcat = StartMeshcat()

def build_env():
    """Load in models and build the diagram."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/pr2_shelves.dmd.yaml")
    plant.Finalize()

    MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph.get_query_output_port(),
        meshcat,
        MeshcatVisualizerParams(delete_on_initialization_event=False),
    )

    diagram = builder.Build()
    return diagram, plant, scene_graph

def filterCollsionGeometry(scene_graph, context=None):
    """Some robot models may appear to have self collisions due to overlapping collision geometries.
    This function filters out such problems for our PR2 model."""
    if context is None:
        filter_manager = scene_graph.collision_filter_manager()
    else:
        filter_manager = scene_graph.collision_filter_manager(context)
    inspector = scene_graph.model_inspector()

    pr2 = {}
    shelves = []
    tables = []

    for gid in inspector.GetGeometryIds(
        GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
    ):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))
        if "pr2" in gid_name:
            link_name = gid_name.split("::")[1]
            pr2[link_name] = [gid]

    def add_exclusion(set1, set2=None):
        if set2 is None:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
            )
        else:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    GeometrySet(set1), GeometrySet(set2)
                )
            )

    # Robot-to-self collisions
    add_exclusion(
        pr2["base_link"],
        pr2["l_shoulder_pan_link"]
        + pr2["r_shoulder_pan_link"]
        + pr2["l_upper_arm_link"]
        + pr2["r_upper_arm_link"]
        + pr2["head_pan_link"]
        + pr2["head_tilt_link"],
    )
    add_exclusion(
        pr2["torso_lift_link"], pr2["head_pan_link"] + pr2["head_tilt_link"]
    )
    add_exclusion(
        pr2["l_shoulder_pan_link"] + pr2["torso_lift_link"],
        pr2["l_upper_arm_link"],
    )
    add_exclusion(
        pr2["r_shoulder_pan_link"] + pr2["torso_lift_link"],
        pr2["r_upper_arm_link"],
    )
    add_exclusion(pr2["l_forearm_link"], pr2["l_gripper_palm_link"])
    add_exclusion(pr2["r_forearm_link"], pr2["r_gripper_palm_link"])

goal_rotation = RotationMatrix(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]
)
goal_position = np.array([-0.83, 0.18, 1.4])
goal_pose = RigidTransform(goal_rotation, goal_position)

def solve_ik(X_WG, max_tries=10, fix_base=False, base_pose=np.zeros(3)):
    diagram, plant, scene_graph = build_env()

    gripper_frame = plant.GetFrameByName("l_gripper_palm_link")

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)
    filterCollsionGeometry(scene_graph, sg_context)

    # Note: passing in a plant_context is necessary for collision-free constraints!
    ik = InverseKinematics(plant, plant_context)
    q_variables = ik.q()  # Get variables for MathematicalProgram
    prog = ik.prog()  # Get MathematicalProgram
    q_nominal = np.zeros(len(q_variables))
    q_nominal[0:3] = base_pose
    prog.AddQuadraticErrorCost(
        np.eye(len(q_variables)), q_nominal, q_variables
    )

    # Add your constraints here    

    # adding distance constraint
    ik.AddMinimumDistanceLowerBoundConstraint(0.01)
    # print(q_variables)
    gripper_pose = X_WG.translation()
    threshold_gripper = 0.001

    # Order of variables is arbitrary (decided by us!)
    # I have joint positions and want to get the end effector position

    # keep the constraint in the WG frame
    # we want to check whether we're in the right frame
    ik.AddOrientationConstraint(
        frameAbar = plant.world_frame(), 
        R_AbarA = X_WG.rotation(), 
        frameBbar = gripper_frame, 
        R_BbarB = RotationMatrix(), 
        theta_bound=0.01
    )
    # ik.AddPositionConstraint(X_WG.translation() - 0.1, X_WG.translation() + 0.1)
    ik.AddPositionConstraint(
        frameA = plant.world_frame(), frameB = gripper_frame, p_BQ = np.zeros(3), 
        p_AQ_lower = X_WG.translation() - 0.001, p_AQ_upper = X_WG.translation() + 0.001
    )

    # TODO: ADD ROTATION CONSTRAINTS!!!

    if fix_base:
        # print(q_nominal)
        prog.AddConstraint(q_variables[0] == base_pose[0])
        prog.AddConstraint(q_variables[1] == base_pose[1])
        prog.AddConstraint(q_variables[2] == base_pose[2])
        # TODO: ADD ROTATION CONSTRAINT!
    

    for count in range(max_tries):
        # Compute a random initial guess here
        # TODO: GET THE RANDOM INITIAL GUESS!!
        guess = []
        for low, high in zip(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()):
            if low == -float("inf"):
                low = -np.pi
            if high == float("inf"):
                high = np.pi
            guess.append(np.random.uniform(low, high))
        prog.SetInitialGuess(q_variables, np.array(guess))

        result = Solve(prog)

        if running_as_notebook:
            render_context = diagram.CreateDefaultContext()
            plant.SetPositions(
                plant.GetMyContextFromRoot(render_context),
                result.GetSolution(q_variables),
            )
            diagram.ForcedPublish(context)

        if result.is_success():
            print("Succeeded in %d tries!" % (count + 1))
            return result.GetSolution(q_variables)

    print("Failed!")
    return None

solve_ik(
    goal_pose,
    max_tries=20,
    fix_base=True,
    base_pose=np.array([-1.23, 0.05, 0]),
)

solve_ik(goal_pose, fix_base=False)

input("Press any key to stop the simulation! ")
