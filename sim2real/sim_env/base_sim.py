import argparse
import sys
import threading
import time
from threading import Thread
import os

# ---------- Headless / Window switch ----------
HEADLESS = os.getenv("HEADLESS", "1") == "1"
if HEADLESS:
    # Fully bypass X/GLFW and use off-screen
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("MUJOCO_GL", "egl")         # or "osmesa" if no GPU
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy") # if pygame used elsewhere
else:
    # Windowed mode: ensure we do NOT force EGL/OSMesa here
    os.environ.pop("MUJOCO_GL", None)

import mujoco
# Only import viewer in windowed mode
if not HEADLESS:
    import mujoco.viewer
import numpy as np
import yaml
from loguru import logger
from loop_rate_limiters import RateLimiter

import imageio
import matplotlib
if HEADLESS:
    matplotlib.use("Agg")  # no GUI backend
import matplotlib.pyplot as plt

sys.path.append("../")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from sim2real.utils.robot import Robot

from sim2real.utils.sdk2py_bridge import ElasticBand, create_sdk2py_bridge


class BaseSimulator:
    def __init__(self, config):
        self.config = config

        self.force_body_names = ["left_elbow_link", "right_elbow_link"]  # 你要施加力的 link 名
        self.force_body_id = [None] * len(self.force_body_names)  # 初始化
        self.force_enabled = False  # 新增
        self.force_start_time = 15.0  # 10秒后施加力

        self.init_config()
        self.init_scene()
        self.init_factory()
        self.init_robot_bridge()

        self.renderer = None
        self.video_writer = None
        self.record_video = False  # 控制是否录制
        self.video_stop = False
        self.video_path = "record_video.mp4"
        self.record_start_time = 15.0  # 10秒后开始录制
        self.record_end_time = 30.0    # 20秒后结束录制

        self.sim_thread = Thread(target=self.simulation_thread, name="sim_thread", daemon=True)

    def init_config(self):
        self.robot = Robot(self.config)
        self.sdk_type = self.config.get("SDK_TYPE", "unitree")
        self.num_dof = self.robot.NUM_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_dof)
        self.logger = logger
        self.rate = RateLimiter(1 / self.config["SIMULATE_DT"])

    def init_factory(self):
        if self.sdk_type == "unitree":
            if self.config.get("INTERFACE", None):
                if sys.platform == "linux":
                    self.config["INTERFACE"] = "lo"
                elif sys.platform == "darwin":
                    self.config["INTERFACE"] = "lo0"
                else:
                    raise NotImplementedError("Only support Linux and MacOS.")
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        elif self.sdk_type == "booster":
            from booster_robotics_sdk_python import ChannelFactory

            ChannelFactory.Instance().Init(self.config["DOMAIN_ID"])
        else:
            raise NotImplementedError(f"SDK type {self.sdk_type} is not supported yet")
        self.logger.info(str.format("SDK TYPE: {0}", self.sdk_type))

    def init_scene(self):
        print(self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

        base_body_name = self.config.get("BASE_BODY_NAME", "pelvis")
        self.base_id = self.mj_model.body(base_body_name).id

        self.force_body_ids = [self.mj_model.body(name).id for name in self.force_body_names]

        # Viewer only in windowed mode
        if not HEADLESS:
            # Enable the elastic band
            if self.config["ENABLE_ELASTIC_BAND"]:
                self.elastic_band = ElasticBand()
                band_attached_link_name = self.config.get("BAND_ATTACHED_LINK", "torso_link")
                self.band_attached_link = self.mj_model.body(band_attached_link_name).id
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback
                )
            else:
                self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        else:
            self.viewer = None  # no window

        # Cache camera names for safe selection
        self.camera_names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                             for i in range(self.mj_model.ncam)]
        self.pref_camera = "track" if "track" in self.camera_names else None

    def init_robot_bridge(self):
        self.robot_bridge = create_sdk2py_bridge(self.mj_model, self.mj_data, self.config)
        if self.config["USE_JOYSTICK"]:
            if sys.platform == "linux" and self.config["SDK_TYPE"] == "unitree":
                self.robot_bridge.SetupJoystick(
                    device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
                )
            else:
                self.logger.warning("Joystick not supported on this platform/SDK.")

    def compute_torques(self):
        if self.robot_bridge.low_cmd:
            motor_cmd = list(self.robot_bridge.low_cmd.motor_cmd)
            try:
                for i in range(self.robot_bridge.num_motor):
                    self.torques[i] = (
                        motor_cmd[i].tau
                        + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[7 + i])
                        + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[6 + i])
                    )
            except Exception as e:
                self.logger.error(str.format("Joint {0} not found in motor_cmd: {1}", i, e))
        # Set the torque limit
        self.torques = np.clip(self.torques, -self.robot_bridge.torque_limit, self.robot_bridge.torque_limit)

    def _ensure_renderer(self):
        """Create renderer in this (simulation) thread."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        if self.video_writer is None:
            # stream writer to avoid huge RAM usage
            self.video_writer = imageio.get_writer(self.video_path, fps=int(1 / self.sim_dt))

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()
        if self.config.get("ENABLE_ELASTIC_BAND", False) and hasattr(self, "elastic_band") and self.elastic_band.enable:
            self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                self.mj_data.qpos[:3], self.mj_data.qvel[:3]
            )
        # # 只在仿真时间大于15秒后施加力
        # if self.mj_data.time >= self.force_start_time:
        #     for body_id in self.force_body_ids:
        #         r = np.array([0.25, 0.0, 0.0])  # 偏移向量，可为每个 link 单独设置
        #         # F = np.array([0.0, 0.0, -30.0])  # 施加的力，可为每个 link 单独设置
        #         # 2秒周期的正弦力
        #         F_amp = 30.0
        #         period = 4.0
        #         force = -F_amp * np.sin(2 * np.pi * self.mj_data.time / period)
        #         F = np.array([0.0, 0.0, force])
        #         torque = np.cross(r, F)
        #         wrench = np.concatenate([F, torque])
        #         self.mj_data.xfrc_applied[body_id, :] = wrench
        # else:
        #     for body_id in self.force_body_ids:
        #         self.mj_data.xfrc_applied[body_id, :] = np.zeros(6)

        self.compute_torques()
        if self.robot_bridge.free_base:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        # 仿真时间大于15秒且小于等于20秒时录制视频
        self.record_video = (self.record_start_time <= self.mj_data.time <= self.record_end_time)

        # print("Available cameras:", [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i) 
        #                     for i in range(self.mj_model.ncam)])

        # 渲染并保存帧
        if self.record_video:
            if self.renderer is None:
                self._ensure_renderer()
            cam = self.pref_camera  # None means free camera
            self.renderer.update_scene(self.mj_data, camera=cam)
            frame = self.renderer.render()
            self.video_writer.append_data(frame)

    def _window_loop_condition(self):
        return (self.viewer is not None) and self.viewer.is_running()

    def simulation_thread(self):
        sim_cnt = 0
        start_time = time.time()

        # If headless and you still want to watch progress, lazy-create renderer now
        if HEADLESS:
            self._ensure_renderer()

        while True:
            if not HEADLESS:
                if not self._window_loop_condition():
                    break

            self.sim_step()

            if not HEADLESS:
                # sync window at viewer_dt
                if sim_cnt % max(1, int(self.viewer_dt / self.sim_dt)) == 0:
                    self.viewer.sync()
            # Get FPS
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                self.logger.info(str.format("FPS: {0:.2f}", 100 / (end_time - start_time)))
                start_time = end_time

            self.rate.sleep()
            # 仿真结束后写视频
            if (self.mj_data.time >= self.record_end_time) and (self.video_writer is not None):
                self.video_writer.close()
                self.logger.info(f"Video saved to {self.video_path}")
                self.video_writer = None
                break  # end demo loop; remove if you want continuous sim

        # Clean up renderer (optional)
        self.renderer = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = BaseSimulator(config)
    simulation.sim_thread.start()
    simulation.sim_thread.join()
