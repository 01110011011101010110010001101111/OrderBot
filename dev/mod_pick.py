import numpy as np
from pydrake.all import (
    AngleAxis,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
)


def MakeGripperFrames_Squeeze(X_G, t0=0):
    # we instead want it to take in the sqeeze position
    # it should go pick -> squeeze -> pick (pick = place in this case)
    """
    Takes a partial specification with X_G["initial"], X_G["pick"], and
    X_G["place"], and returns a X_G and times with all of the pick and place
    frames populated.
    """
    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.2, 0])
    X_GgraspGpregrasp_post = RigidTransform([0, -0.3, 0])

    X_G["prepick"] = X_G["pick"] @ X_GgraspGpregrasp
    # X_G["prepick"].set_rotation(RotationMatrix.MakeZRotation(np.pi / 2))
    X_G["preplace"] = X_G["place"] @ X_GgraspGpregrasp_post 
    # X_G["preplace"].set_rotation(RotationMatrix.MakeZRotation(-np.pi / 3))

    # I'll interpolate a halfway orientation by converting to axis angle and
    # halving the angle.
    X_GinitialGprepick = X_G["initial"].inverse() @ X_G["prepick"]
    angle_axis = X_GinitialGprepick.rotation().ToAngleAxis()
    X_GinitialGprepare = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GinitialGprepick.translation() / 2.0,
    )
    X_G["prepare"] = X_G["initial"] @ X_GinitialGprepare
    p_G = np.array(X_G["prepare"].translation())
    p_G[2] = 0.5
    # To avoid hitting the cameras, make sure the point satisfies x - y < .5
    if p_G[0] - p_G[1] < 0.5:
        scale = 0.5 / (p_G[0] - p_G[1])
        p_G[:1] /= scale
    X_G["prepare"].set_translation(p_G)

    X_GprepickGpreplace = X_G["prepick"].inverse() @ X_G["preplace"]
    angle_axis = X_GprepickGpreplace.rotation().ToAngleAxis()
    X_GprepickGclearance = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GprepickGpreplace.translation() / 2.0,
    )
    X_G["clearance"] = X_G["prepick"] @ X_GprepickGclearance
    p_G = np.array(X_G["clearance"].translation())
    p_G[2] = 0.5
    # To avoid hitting the cameras, make sure the point satisfies x - y < .5
    if p_G[0] - p_G[1] < 0.5:
        scale = 0.5 / (p_G[0] - p_G[1])
        p_G[:1] /= scale
    X_G["clearance"].set_translation(p_G)
    print(X_G["clearance"].rotation())
    X_G["clearance2"] = X_G["clearance"]

    # Now let's set the timing
    times = {"initial": t0}
    prepare_time = 10.0 * np.linalg.norm(X_GinitialGprepare.translation())
    times["prepare"] = times["initial"] + prepare_time
    times["prepick"] = times["prepare"] + prepare_time
    # Allow some time for the gripper to close.
    times["pick_start"] = times["prepick"] + 2.0
    times["pick_end"] = times["pick_start"] + 2.0
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["postpick"] = times["pick_end"] + 2.0
    X_G["postpick"] = X_G["prepick"]
    time_to_from_clearance = 10.0 * np.linalg.norm(
        X_GprepickGclearance.translation()
    )
    times["clearance"] = times["postpick"] + time_to_from_clearance
    times["preplace"] = times["clearance"] + time_to_from_clearance
    times["preplace_start"] = times["preplace"] + 2.0
    times["preplace_hold"] = times["preplace_start"] + 1.0
    times["preplace_stop"] = times["preplace_hold"] + 2.0
    times["clearance2"] = times["preplace_stop"] + time_to_from_clearance
    times["place_start"] = times["clearance2"] + 2.0
    times["place_end"] = times["place_start"] + 2.0
    X_G["clearance2"] = RigidTransform() # X_G["pick"].copy()
    X_G["place"] = X_G["pick"]
    X_G["preplace_stop"] = X_G["preplace"]
    X_G["preplace_start"] = X_G["preplace"]
    X_G["preplace_hold"] = X_G["preplace"]
    p_G = np.array(X_G["pick"].translation())
    p_G[2] += 0.25
    X_G["clearance2"].set_translation(p_G)
    X_G["clearance2"].set_rotation(X_G["pick"].rotation())
    X_G["place_start"] = X_G["pick_start"]
    X_G["place_end"] = X_G["pick"]
    # X_G["place_start"] = X_G["place"]
    # X_G["place_end"] = X_G["place"]
    times["clearance3"] = times["place_end"] + 2.0
    X_G["clearance3"] = X_G["clearance2"]
    times["postplace"] = times["clearance3"] + 2.0
    X_G["postplace"] = X_G["preplace"]

    return X_G, times


def MakeGripperPoseTrajectory_Squeeze(X_G, times):
    """Constructs a gripper position trajectory from the plan "sketch"."""
    sample_times = []
    poses = []
    for name in [
        "initial",
        "prepare",
        "prepick",
        "pick_start",
        "pick_end",
        "postpick",
        "clearance",
        "preplace",
        "preplace_start",
        "preplace_hold",
        "preplace_stop",
        "clearance2",
        "place_start",
        "place_end",
        "clearance3",
        "postplace",
    ]:
        sample_times.append(times[name])
        poses.append(X_G[name])

    return PiecewisePose.MakeLinear(sample_times, poses)


# TODO: MAKE THIS NOT OPEN THE GRIPPER AT THE END
# ALTERNATIVELY, ADD SEVERAL EXTRA STEPS IN BETWEEN
def MakeGripperCommandTrajectory_Squeeze(times):
    """Constructs a WSG command trajectory from the plan "sketch"."""
    opened = np.array([0.107])
    closed = np.array([0.03])
    # NOTE: might mess with some of the GO_HOME logic
    squeeze = np.array([0.0])

    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["pick_start"]],
        np.hstack([[opened], [opened]]),
    )
    traj_wsg_command.AppendFirstOrderSegment(times["pick_end"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["postpick"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["clearance"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["preplace"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["preplace_start"], squeeze)
    traj_wsg_command.AppendFirstOrderSegment(times["preplace_hold"], squeeze)
    traj_wsg_command.AppendFirstOrderSegment(times["preplace_stop"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["clearance2"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_start"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_end"], opened)
    traj_wsg_command.AppendFirstOrderSegment(times["clearance3"], opened)
    traj_wsg_command.AppendFirstOrderSegment(times["postplace"], opened)
    return traj_wsg_command

