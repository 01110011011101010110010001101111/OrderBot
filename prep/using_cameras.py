import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pydot
from IPython.display import HTML, SVG, display
from matplotlib.patches import Rectangle
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    Quaternion,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
)

from manipulation import running_as_notebook
from manipulation.mustard_depth_camera_example import MustardExampleSystem

# Start the visualizer.
meshcat = StartMeshcat()

class SimpleCameraSystem:
    def __init__(self):
        diagram = MustardExampleSystem()
        context = diagram.CreateDefaultContext()

        # setup
        meshcat.SetProperty("/Background", "visible", False)

        # getting data
        self.point_cloud = diagram.GetOutputPort("camera0_point_cloud").Eval(
            context
        )
        self.depth_im_read = (
            diagram.GetOutputPort("camera0_depth_image")
            .Eval(context)
            .data.squeeze()
        )
        self.depth_im = deepcopy(self.depth_im_read)
        self.depth_im[self.depth_im == np.inf] = 10.0
        label_im = (
            diagram.GetOutputPort("camera0_label_image")
            .Eval(context)
            .data.squeeze()
        )
        self.rgb_im = (
            diagram.GetOutputPort("camera0_rgb_image").Eval(context).data
        )
        self.mask = label_im == 1

        # draw visualization
        meshcat.SetObject("point_cloud", self.point_cloud)

        # camera specs
        cam0 = diagram.GetSubsystemByName("camera0")
        cam0_context = cam0.GetMyMutableContextFromRoot(context)
        self.X_WC = cam0.GetOutputPort("body_pose_in_world").Eval(cam0_context)
        self.X_WC = RigidTransform(self.X_WC)  # See drake issue #15973
        self.cam_info = cam0.depth_camera_info()

        # get points for mustard bottle
        depth_mustard = self.mask * self.depth_im
        u_range = np.arange(depth_mustard.shape[0])
        v_range = np.arange(depth_mustard.shape[1])
        depth_v, depth_u = np.meshgrid(v_range, u_range)
        depth_pnts = np.dstack([depth_u, depth_v, depth_mustard])
        depth_pnts = depth_pnts.reshape(
            [depth_pnts.shape[0] * depth_pnts.shape[1], 3]
        )
        pC = self.project_depth_to_pC(depth_pnts)
        p_C_mustard = pC[pC[:, 2] > 0]
        self.p_W_mustard = self.X_WC.multiply(p_C_mustard.T).T

    def get_color_image(self):
        return deepcopy(self.rgb_im[:, :, 0:3])

    def get_intrinsics(self):
        # read camera intrinsics
        cx = self.cam_info.center_x()
        cy = self.cam_info.center_y()
        fx = self.cam_info.focal_x()
        fy = self.cam_info.focal_y()
        return cx, cy, fx, fy

    def project_depth_to_pC(self, depth_pixel):
        """
        project depth pixels to points in camera frame
        using pinhole camera model
        Input:
            depth_pixels: numpy array of (nx3) or (3,)
        Output:
            pC: 3D point in camera frame, numpy array of (nx3)
        """
        # switch u,v due to python convention
        v = depth_pixel[:, 0]
        u = depth_pixel[:, 1]
        Z = depth_pixel[:, 2]
        cx, cy, fx, fy = self.get_intrinsics()
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pC = np.c_[X, Y, Z]
        return pC


def bbox(img):
    a = np.where(img != 0)
    bbox = ([np.min(a[0]), np.max(a[0])], [np.min(a[1]), np.max(a[1])])
    return bbox


env = SimpleCameraSystem()
X_WC = env.X_WC
p_W_mustard = env.p_W_mustard
K = env.cam_info.intrinsic_matrix()
rgb_im = env.get_color_image()


def deproject_pW_to_image(p_W_mustard, cx, cy, fx, fy, X_WC):
    """
    convert points in the world frame to camera pixels
    Input:
        - p_W_mustard: points of the mustard bottle in world frame (nx3)
        - fx, fy, cx, cy: camera intrinsics
        - X_WC: camera pose in the world frame
    Output:
        - mask: numpy array of size 480x640
    """
    p_C_mustard = X_WC.inverse().multiply(p_W_mustard.T).T

    X = p_C_mustard[:, 0]
    Y = p_C_mustard[:, 1]
    Z = p_C_mustard[:, 2]
    u = (X * fx / Z) + cx
    v = (Y * fy) / Z + cy
    res = np.c_[v, u, Z]

    mask = np.zeros([480, 640])
    for vi, ui, zi in zip(v, u, Z):
        vi = round(vi)
        ui = round(ui)
        mask[vi:vi+1, ui:ui+1] = True
    return mask

cx, cy, fx, fy = env.get_intrinsics()
mask = deproject_pW_to_image(p_W_mustard, cx, cy, fx, fy, X_WC)
print(mask)
plt.imshow(mask)
plt.show()
plt.imshow(rgb_im)

# cx, cy, fx, fy = env.get_intrinsics()
# mask = deproject_pW_to_image(p_W_mustard, cx, cy, fx, fy, X_WC)
# plt.imshow(mask)

