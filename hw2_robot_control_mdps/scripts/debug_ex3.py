"""Headless entry point so the debugger can stop inside the ex3 TODOs.

SO100TrackEnv calls these functions from reset, step, and the observation.
Building that env calls get_obs immediately, and scripts/train.py starts many
processes. This driver calls each function with the same inputs the env uses.
"""
import mujoco
import numpy as np

from __init__ import XML_PATH
from exercises.ex3 import (
    compute_reward,
    get_obs,
    process_action,
    reset_robot,
    reset_target_position,
)

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    default_qpos = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])
    reset_qpos = reset_robot(default_qpos)
    data.qpos[:] = reset_qpos
    mujoco.mj_forward(model, data)

    base_pos = data.body("Base").xpos.copy()
    target_pos = reset_target_position(base_pos)

    action = np.zeros(6, dtype=np.float64)
    target_qpos = process_action(action, model.jnt_range)

    reward = compute_reward(0.01)

    obs = get_obs(
        data.qpos.flat[:].copy(),
        data.site("ee_site").xpos.copy(),
        data.site("ee_site").xmat.reshape(3, 3).copy(),
        data.body("Base").xpos.copy(),
        data.body("Base").xmat.reshape(3, 3).copy(),
        np.asarray(target_pos, dtype=np.float64),
    )
    print(reset_qpos, target_pos, target_qpos, reward, obs.shape)
