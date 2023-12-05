import gym
import numpy as np
import torch
from pydrake.all import StartMeshcat

from manipulation.utils import LoadDataResource, running_as_notebook
from manipulation.envs.box_flipup import BoxFlipUpEnv

from psutil import cpu_count

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

num_cpu = int(cpu_count() / 2) if running_as_notebook else 2

# Optional imports (these are heavy dependencies for just this one notebook)
sb3_available = False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env

    sb3_available = True
except ImportError:
    print("stable_baselines3 not found")
    print("Consider 'pip3 install stable_baselines3'.")


meshcat = StartMeshcat()

# TODO(russt): Move this to box_flipup.py?
gym.envs.register(
    id="BoxFlipUp-v0", entry_point="manipulation.envs.box_flipup:BoxFlipUpEnv"
)

observations = "state"
time_limit = 10 if running_as_notebook else 0.5

# Note: Models saved in stable baselines are version specific.  This one
# requires python3.8 (and cloudpickle==1.6.0).
zip = f"box_flipup_ppo_{observations}.zip"

env = make_vec_env(
    BoxFlipUpEnv,
    n_envs=num_cpu,
    seed=0,
    vec_env_cls=SubprocVecEnv,
    vec_env_kwargs={
        "start_method": "fork",
    },
    env_kwargs={
        "observations": observations,
        "time_limit": time_limit,
    },
)

use_pretrained_model = True
if use_pretrained_model:
    # TODO(russt): Save a trained model that works on Deepnote.
    model = PPO.load(LoadDataResource(zip), env)
elif running_as_notebook:
    # This is a relatively small amount of training.  See rl/train_boxflipup.py
    # for a version that runs the heavyweight version with multiprocessing.
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
else:
    # For testing this notebook, we simply want to make sure that the code runs.
    model = PPO("MlpPolicy", env, n_steps=4, n_epochs=2, batch_size=8)
    model.learn(total_timesteps=4)

# Make a version of the env with meshcat.
env = gym.make("BoxFlipUp-v0", meshcat=meshcat, observations=observations)

if running_as_notebook:
    env.simulator.set_target_realtime_rate(1.0)

obs = env.reset()
for i in range(500 if running_as_notebook else 5):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

obs = env.reset()
Q, Qdot = np.meshgrid(np.arange(0, np.pi, 0.05), np.arange(-2, 2, 0.05))
# TODO(russt): tensorize this...
V = 0 * Q
for i in range(Q.shape[0]):
    for j in range(Q.shape[1]):
        obs[2] = Q[i, j]
        obs[7] = Qdot[i, j]
        with torch.no_grad():
            V[i, j] = (
                model.policy.predict_values(
                    model.policy.obs_to_tensor(obs)[0]
                )[0]
                .cpu()
                .numpy()[0]
            )
V = V - np.min(np.min(V))
V = V / np.max(np.max(V))

meshcat.Delete()
meshcat.ResetRenderMode()
plot_surface(meshcat, "Critic", Q, Qdot, V, wireframe=True)

import pydot
from IPython.display import display, SVG

env = BoxFlipUpEnv()
display(
    SVG(
        pydot.graph_from_dot_data(
            env.simulator.get_system().GetGraphvizString(max_depth=1)
        )[0].create_svg()
    )
)

input("")
