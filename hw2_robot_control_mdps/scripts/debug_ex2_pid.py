"""Headless entry point so the debugger can stop inside pid_control.

scripts/pid_control.py calls pid_control from the MuJoCo control callback,
after launch_passive. On macOS that viewer only runs under mjpython, so this
driver builds the same error history and calls pid_control directly.
"""
import mujoco
import numpy as np

from __init__ import TORQUE_CTRL_XML_PATH
from exercises.ex2 import pid_control

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(TORQUE_CTRL_XML_PATH))
    data = mujoco.MjData(model)

    current_qpos = data.qpos.copy()
    target_qpos = current_qpos + 0.1
    error = target_qpos - current_qpos
    # Two samples, as update_tracking_error_history stacks rows of joint error.
    tracking_error_history = np.vstack([error, 0.5 * error])

    control = pid_control(tracking_error_history, model.opt.timestep)
    print(control)
