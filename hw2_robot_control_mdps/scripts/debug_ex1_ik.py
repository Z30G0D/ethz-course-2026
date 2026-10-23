"""Headless entry point so the debugger can stop inside ik_track.

The MuJoCo viewer in inverse_kinematics.py uses launch_passive, which on macOS
only runs under mjpython. This driver calls the same IK loop without opening
a window.
"""
import mujoco

from __init__ import XML_PATH
from exercises.ex1 import build_keypoints, ik_track

if __name__ == "__main__":
    keypoints = build_keypoints()
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    ik_track(model, data, "ee_site", keypoints[0], dt=0.1)
