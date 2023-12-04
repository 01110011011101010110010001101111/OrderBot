import logging
from copy import copy
from enum import Enum

import numpy as np
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    InputPortIndex,
    LeafSystem,
    MeshcatVisualizer,
    Parser,
    PiecewisePolynomial,
    PiecewisePose,
    PointCloud,
    PortSwitch,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
)

from manipulation import ConfigureParser, FindResource, running_as_notebook
from manipulation.clutter import GenerateAntipodalGraspCandidate
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.pick import (
    MakeGripperCommandTrajectory,
    MakeGripperFrames,
    MakeGripperPoseTrajectory,
)
from manipulation.scenarios import AddIiwaDifferentialIK, ycb
from manipulation.station import (
    AddPointClouds,
    MakeHardwareStation,
    add_directives,
    load_scenario,
)

from kinematics import GraspSelector

full_path = "/Users/paromitadatta/Desktop/64210/6.4210-Final-Project/objects/"


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")


logging.getLogger("drake").addFilter(NoDiffIKWarnings())

# Start the visualizer.
meshcat = StartMeshcat()

rng = np.random.default_rng(135)  # this is for python
generator = RandomGenerator(rng.integers(0, 1000))  # this is for c++

# overridding the running_as_a_notebook
running_as_notebook = True

tasks = ["bread", "chicken", "bread"]
so_far = [False, False, False]
idx = -1

states = {
    "bread": 3,
    "chicken": 2,
}

mode_to_str = {states[key]: key for key in states}

# States for state machine
class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    GO_HOME = 2
    PICKING_FROM_X_BIN = 2
    PICKING_FROM_Y_BIN = 3
    PICKING_FROM_Z_BIN = 5


class Planner(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self._x_bin_grasp_index = self.DeclareAbstractInputPort(
            "x_bin_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        self._y_bin_grasp_index = self.DeclareAbstractInputPort(
            "y_bin_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        # self._z_bin_grasp_index = self.DeclareAbstractInputPort(
        #     "z_bin_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        # ).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort("wsg_state", 2).get_index()

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
        )
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._attempts_index = self.DeclareDiscreteState(1)

        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        # For GoHome mode.
        num_positions = 7
        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa_position", num_positions
        ).get_index()
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )
        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset,
        )
        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self.DeclareVectorOutputPort(
            "iiwa_position_command", num_positions, self.CalcIiwaPosition
        )
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if context.get_time() - times["initial"] > 1.0:
                self.Plan(context, state)
            return
        elif mode == PlannerState.GO_HOME:
            traj_q = context.get_mutable_abstract_state(
                int(self._traj_q_index)
            ).get_value()
            if not traj_q.is_time_in_range(current_time):
                self.Plan(context, state)
            return

        # If we are between pick and place and the gripper is closed, then
        # we've missed or dropped the object.  Time to replan.
        if current_time > times["postpick"] and current_time < times["preplace"]:
            wsg_state = self.get_input_port(self._wsg_state_index).Eval(context)
            if wsg_state[0] < 0.01:
                attempts = state.get_mutable_discrete_state(
                    int(self._attempts_index)
                ).get_mutable_value()
                if attempts[0] > 5:
                    # If I've failed 5 times in a row, then switch bins.
                    print(
                        "Switching to the other bin after 5 consecutive failed attempts"
                    )
                    attempts[0] = 0
                    if mode == PlannerState.PICKING_FROM_X_BIN:
                        state.get_mutable_abstract_state(
                            int(self._mode_index)
                        ).set_value(PlannerState.PICKING_FROM_Y_BIN)
                    else:
                        state.get_mutable_abstract_state(
                            int(self._mode_index)
                        ).set_value(PlannerState.PICKING_FROM_X_BIN)
                    self.Plan(context, state)
                    return

                attempts[0] += 1
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                    PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE
                )
                times = {"initial": current_time}
                state.get_mutable_abstract_state(int(self._times_index)).set_value(
                    times
                )
                X_G = self.get_input_port(0).Eval(context)[
                    int(self._gripper_body_index)
                ]
                state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
                    PiecewisePose.MakeLinear([current_time, np.inf], [X_G, X_G])
                )
                return

        traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
        if not traj_X_G.is_time_in_range(current_time):
            self.Plan(context, state)
            return

        X_G = self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        # if current_time > 10 and current_time < 12:
        #    self.GoHome(context, state)
        #    return
        if (
            np.linalg.norm(
                traj_X_G.GetPose(current_time).translation() - X_G.translation()
            )
            > 0.2
        ):
            # If my trajectory tracking has gone this wrong, then I'd better
            # stop and replan.  TODO(russt): Go home, in joint coordinates,
            # instead.
            self.GoHome(context, state)
            return

    def GoHome(self, context, state):
        print("Replanning due to large tracking error.")
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            PlannerState.GO_HOME
        )
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[0] = q[0]  # Safer to not reset the first joint.

        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 5.0], np.vstack((q, q0)).T
        )
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(q_traj)

    def Plan(self, context, state):
        mode = copy(state.get_mutable_abstract_state(int(self._mode_index)).get_value())

        X_G = {
            "initial": self.get_input_port(0).Eval(context)[
                int(self._gripper_body_index)
            ]
        }

        global tasks, so_far, states, idx
        idx += 1
        if idx > len(tasks):
            assert False
        mode = states[tasks[idx]]

        # TODO: CAN MODIFY TO WORK WITH DIFFERENT BINS WITH DIFFERENT THINGS
        cost = np.inf
        retry = False
        for i in range(5):
            # Y == chicken
            # X == bread

            # print(PlannerState, got_first_bread, got_second_bread, got_filling)

            # right now, default to if tried, succeeded. Can update later
            if mode == PlannerState.PICKING_FROM_Y_BIN:
                if retry:
                    # if we have to retry, don't add any new logic
                    cost, X_G["pick"] = self.get_input_port(self._y_bin_grasp_index).Eval(
                        context
                    )
                else:
                    cost, X_G["pick"] = self.get_input_port(self._x_bin_grasp_index).Eval(
                        context
                    )
                    got_filling = True
                    mode = PlannerState.PICKING_FROM_X_BIN
            elif mode == PlannerState.PICKING_FROM_X_BIN:
                # if we have to retry, don't add any new logic
                if retry: 
                    cost, X_G["pick"] = self.get_input_port(self._x_bin_grasp_index).Eval(
                        context
                    )
                    break
                if got_first_bread:
                    # we're done
                    mode = PlannerState.GO_HOME
                    got_second_bread = True
                    assert False, "Done!"
                else:
                    got_first_bread = True
                    mode = PlannerState.PICKING_FROM_Y_BIN
                    cost, X_G["pick"] = self.get_input_port(self._y_bin_grasp_index).Eval(
                        context
                    )
 
            else:
                mode = states[tasks[idx]]
                # idx += 1
                cost, X_G["pick"] = self.get_input_port(self._x_bin_grasp_index).Eval(
                    context
                )
 
                    
            # if mode == PlannerState.PICKING_FROM_Y_BIN:
            #     cost, X_G["pick"] = self.get_input_port(self._y_bin_grasp_index).Eval(
            #         context
            #     )
            #     if np.isinf(cost):
            #         mode = PlannerState.PICKING_FROM_X_BIN
            # else:
            #     # cost, X_G["pick"] = self.get_input_port(self._x_bin_grasp_index).Eval(
            #     #     context
            #     # )
            #     # if np.isinf(cost):
            #     if True:
            #         mode = PlannerState.PICKING_FROM_Y_BIN
            #     # else:
            #     #     mode = PlannerState.PICKING_FROM_X_BIN

            if not np.isinf(cost):
                break
            else:
                retry = True

        assert not np.isinf(
            cost
        ), "Could not find a valid grasp in either bin after 5 attempts"
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(mode)

        # TODO(russt): The randomness should come in through a random input
        # port.

        # TODO: CAN BE CHANGED TO X AND Y BINS
        
        # placing on the mat...

        # TODO: we'll likely want it to increase as the sandwich size increases
        if mode == PlannerState.PICKING_FROM_X_BIN:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, 0),
                [-0.01, -0.25, 0.15],
                # 0, -0.25, -0.015
            )
        else:
            # making the toppings a bit closer since it bounces as is
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, 0),
                [-0.01, -0.25, 0.12],
                # 0, -0.25, -0.015
            )
 

        # if mode == PlannerState.PICKING_FROM_X_BIN:
        #     # Place in Y bin:
        #     X_G["place"] = RigidTransform(
        #         RollPitchYaw(-np.pi / 2, 0, 0),
        #         [rng.uniform(-0.25, 0.15), rng.uniform(-0.6, -0.4), 0.3],
        #     )
        # else:
        #     # Place in X bin:
        #     X_G["place"] = RigidTransform(
        #         RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
        #         [rng.uniform(0.35, 0.65), rng.uniform(-0.12, 0.28), 0.3],
        #     )

        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode}—{mode_to_str[mode]} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)

        if False:  # Useful for debugging
            AddMeshcatTriad(meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

    def start_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .start_time()
        )

    def end_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index)).get_value().end_time()
        )

    def CalcGripperPose(self, context, output):
        context.get_abstract_state(int(self._mode_index)).get_value()

        traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
        if traj_X_G.get_number_of_segments() > 0 and traj_X_G.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                context.get_abstract_state(int(self._traj_X_G_index))
                .get_value()
                .GetPose(context.get_time())
            )
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(
            self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        )

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        opened = np.array([0.107])
        np.array([0.0])

        if mode == PlannerState.GO_HOME:
            # Command the open position
            output.SetFromVector([opened])
            return

        traj_wsg = context.get_abstract_state(int(self._traj_wsg_index)).get_value()
        if traj_wsg.get_number_of_segments() > 0 and traj_wsg.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return

        # Command the open position
        output.SetFromVector([opened])

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context),
        )

    def CalcIiwaPosition(self, context, output):
        traj_q = context.get_mutable_abstract_state(int(self._traj_q_index)).get_value()

        output.SetFromVector(traj_q.value(context.get_time()))


def clutter_clearing_demo():
    meshcat.Delete()
    builder = DiagramBuilder()

    scenario = load_scenario(
        filename=FindResource(f"{full_path}/_clutter.scenarios.yaml"),
        scenario_name="Clutter",
    )
    model_directives = """
directives:
"""
    NUM_CHICKEN = 10
    for i in range(NUM_CHICKEN):
        # porting over previous work
        ranges = {"x": -0.5, "y": -0.5, "z": -0.05}
        name = "foam_chicken"
        num = i
        model_directives += f"""
- add_model:
    name: {name}_{num}
    file: file://{full_path}{name}.sdf
    default_free_body_pose:
        base_link:
            translation: [{ranges['x'] + np.random.randint(-10, 10)/50}, {ranges['y'] + np.random.randint(-10, 10)/50}, {ranges['z'] + np.random.randint(10)/10}]
"""

    NUM_BREAD = 10
    for num in range(NUM_BREAD):
        ranges = {"x": 0.5, "y": -0.5, "z": -0.05}
        name = "Pound_Cake_OBJ"
        model_directives += f"""
- add_model:
    name: {name}_{num}
    file: file://{full_path}{name}.sdf
    default_free_body_pose:
        base_link:
            translation: [{ranges['x'] + np.random.randint(-10, 10)/50}, {ranges['y'] + np.random.randint(-10, 10)/50}, {ranges['z'] + np.random.randint(10)/10}]
"""
    '''
        model_directives += f"""
- add_model:
    name: {name}_{num}
    file: file://{full_path}{name}.sdf
    default_free_body_pose:
        {name}: # Change here!
            translation: [{0.5 + np.random.randint(-10, 10)/50}, {-0.6 + np.random.randint(-10, 10)/50}, {0.01}] 
            rotation: !Rpy {{ deg: [{np.random.randint(0, 90)}, {np.random.randint(0, 90)}, {np.random.randint(0, 90)}] }}
"""
    '''

    scenario = add_directives(scenario, data=model_directives)

    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder)
    plant = station.GetSubsystemByName("plant")

    ### CREATE GRASP SELECTOR FOR EACH PORT
    y_bin_grasp_selector = builder.AddSystem(
        GraspSelector(
            plant,
            plant.GetModelInstanceByName("bin0"),
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
            ],
        )
    )
    builder.Connect(
        to_point_cloud["camera0"].get_output_port(),
        y_bin_grasp_selector.get_input_port(0),
    )
    builder.Connect(
        to_point_cloud["camera1"].get_output_port(),
        y_bin_grasp_selector.get_input_port(1),
    )
    builder.Connect(
        to_point_cloud["camera2"].get_output_port(),
        y_bin_grasp_selector.get_input_port(2),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        y_bin_grasp_selector.GetInputPort("body_poses"),
    )

    x_bin_grasp_selector = builder.AddSystem(
        GraspSelector(
            plant,
            plant.GetModelInstanceByName("bin1"),
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera3"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera4"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera5"))[0],
            ],
        )
    )
    builder.Connect(
        to_point_cloud["camera3"].get_output_port(),
        x_bin_grasp_selector.get_input_port(0),
    )
    builder.Connect(
        to_point_cloud["camera4"].get_output_port(),
        x_bin_grasp_selector.get_input_port(1),
    )
    builder.Connect(
        to_point_cloud["camera5"].get_output_port(),
        x_bin_grasp_selector.get_input_port(2),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        x_bin_grasp_selector.GetInputPort("body_poses"),
    )

    ## trying to add the z bin

    # z_bin_grasp_selector = builder.AddSystem(
    #     GraspSelector(
    #         plant,
    #         plant.GetModelInstanceByName("bin2"),
    #         camera_body_indices=[
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera6"))[0],
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera7"))[0],
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera8"))[0],
    #         ],
    #     )
    # )
    # builder.Connect(
    #     to_point_cloud["camera6"].get_output_port(),
    #     z_bin_grasp_selector.get_input_port(0),
    # )
    # builder.Connect(
    #     to_point_cloud["camera7"].get_output_port(),
    #     z_bin_grasp_selector.get_input_port(1),
    # )
    # builder.Connect(
    #     to_point_cloud["camera8"].get_output_port(),
    #     z_bin_grasp_selector.get_input_port(2),
    # )
    # builder.Connect(
    #     station.GetOutputPort("body_poses"),
    #     z_bin_grasp_selector.GetInputPort("body_poses"),
    # )

    planner = builder.AddSystem(Planner(plant))
    builder.Connect(
        station.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )
    builder.Connect(
        x_bin_grasp_selector.get_output_port(),
        planner.GetInputPort("x_bin_grasp"),
    )
    builder.Connect(
        y_bin_grasp_selector.get_output_port(),
        planner.GetInputPort("y_bin_grasp"),
    )
    # builder.Connect(
    #     z_bin_grasp_selector.get_output_port(),
    #     planner.GetInputPort("z_bin_grasp"),
    # )
    builder.Connect(
        station.GetOutputPort("wsg.state_measured"),
        planner.GetInputPort("wsg_state"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        planner.GetInputPort("iiwa_position"),
    )

    robot = station.GetSubsystemByName(
        "iiwa.controller"
    ).get_multibody_plant_for_control()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        diff_ik.GetInputPort("robot_state"),
    )
    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"),
        diff_ik.GetInputPort("use_robot_state"),
    )

    builder.Connect(
        planner.GetOutputPort("wsg_position"),
        station.GetInputPort("wsg.position"),
    )

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik"))
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        switch.DeclareInputPort("position"),
    )
    builder.Connect(switch.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        planner.GetOutputPort("control_mode"),
        switch.get_port_selector_input_port(),
    )

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat
    )
    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_context()

    # plant_context = plant.GetMyMutableContextFromRoot(context)
    # z = 0.2
    # for body_index in plant.GetFloatingBaseBodies():
    #     tf = RigidTransform(
    #         UniformlyRandomRotationMatrix(generator),
    #         [rng.uniform(0.35, 0.65), rng.uniform(-0.12, 0.28), z],
    #     )
    #     plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), tf)
    #     z += 0.1

    simulator.AdvanceTo(0.1)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.

    if running_as_notebook:
        simulator.set_target_realtime_rate(1.0)
        meshcat.AddButton("Stop Simulation", "Escape")
        print("Press Escape to stop the simulation")
        while meshcat.GetButtonClicks("Stop Simulation") < 1:
            simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
            pass
        meshcat.DeleteButton("Stop Simulation")


clutter_clearing_demo()

input("Completed Model. Press any key to quit. ")
