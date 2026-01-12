import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from lerobot.cameras.realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import \
    RealSenseCameraConfig
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.config_so100_follower import \
    SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.utils import make_robot_from_config
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2mat
from urchin import URDF

from easyhec import ROBOT_DEFINITIONS_DIR
from easyhec.examples.real.base import Args
from easyhec.utils import visualization
from easyhec.utils.utils_3d import merge_meshes


@dataclass
class SO100Args(Args):
    """Calibrate a (realsense) camera with LeRobot SO100. Note that this script might not work with your particular realsense camera, modify as needed. Other cameras can work if you modify the code to get the camera intrinsics and a single color image from the camera. Results are saved to {output_dir} and organized by the camera name specified in the robot config. Currently only supports off-hand cameras
    
    For your own usage you may have a different camera setup, robot, calibration offsets etc., so we recommend you to copy this file at https://github.com/stonet2000/simple-easyhec/blob/main/easyhec/examples/real/so100.py. 
    
    Before usage make sure to calibrate the robot's motors according to the LeRobot tutorial and look for all comments that start with "CHECK:" which highlight the following:

    1. Check the robot config and make sure the correct camera is used. The default script is for a single realsense camera labelled as "base_camera".
    2. Check and modify the CALIBRATION_OFFSET dictionary to match your own robot's calibration offsets. This is extremely important to tune and is necessary since the 0 degree position of the joints in the real world when calibrated with LeRobot currently do not match the 0 degree position when rendered/simulated.
    3. Modify the initial extrinsic guess if the optimization process fails to converge to a good solution. To save time you can also turn on --use-previous-captures to skip the data collection process if already done once.

    Note that LeRobot SO100 motor calibration is done by moving most joints from one end to another. Make sure to move the joints are far as possible during the LeRobot tutorial on caibration for best results.

    """
    output_dir: str = "results/so100"
    use_previous_captures: bool = False
    """If True, will use the previous collected images and robot segmentations if they exist which can save you time. Otherwise, will prompt you to generate a new segmentation mask. This is useful if you find the initial extrinsic guess is not good enough and simply want to refine that and want to skip the segmentation process."""

    robot_id: Optional[str] = None
    """LeRobot robot ID. If provided will control that robot and will save results to {output_dir}/{robot_id}"""
    realsense_camera_serial_id: str = "146322070293"
    """Realsense camera serial ID."""

# CHECK: This is extrememly important to tune. Run this script with --help for an explanation.
CALIBRATION_OFFSET = {
    "shoulder_pan": 0,
    "shoulder_lift": 0,
    "elbow_flex": 0,
    "wrist_flex": 0,
    "wrist_roll": 0,
    "gripper": 0,
}

# For the author's SO100 they used this calibration offset. Yours might be different
CALIBRATION_OFFSET = {
    "shoulder_pan": -3,
    "shoulder_lift": -3,
    "elbow_flex": 5,
    "wrist_flex": 5,
    "wrist_roll": 0,
    "gripper": 0,
}
CALIBRATION_OFFSET = {
    "shoulder_pan": 0,
    "shoulder_lift": -5,
    "elbow_flex": 5,
    "wrist_flex": -44,
    "wrist_roll": 10,
    "gripper": 10,
}

# CHECK: Check that the created robot config matches the one you wish to use and sets up the port, cameras etc. correctly.
def create_real_robot(uid: str = "so100", robot_id: Optional[str] = None, realsense_serial_number: str = "146322070293") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so100":
        robot_config = SO100FollowerConfig(
            port="/dev/ttyACM0",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            cameras={
                "base_camera": RealSenseCameraConfig(serial_number_or_name=realsense_serial_number, fps=30, width=1280, height=720)
            },
            id=robot_id,
        )
        real_robot = make_robot_from_config(robot_config)
        return real_robot


def main(args: SO100Args):
    user_tuned_calibration_offset = False
    for k in CALIBRATION_OFFSET.keys():
        if CALIBRATION_OFFSET[k] != 0:
            user_tuned_calibration_offset = True
            break
    if not user_tuned_calibration_offset:
        logging.warning("The calibration offset for sim2real/real2sim is not tuned!! Unless you are absolutely sure you will most likely get poor results.")

    robot_id = "default" if args.robot_id is None else args.robot_id
    robot: SO100Follower = create_real_robot("so100", robot_id=args.robot_id, realsense_serial_number=args.realsense_camera_serial_id)
    robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
    robot.connect()

    cameras_ft = robot._cameras_ft
    print(f"Found {len(cameras_ft)} cameras to calibrate")
    for k in cameras_ft.keys():
        (Path(args.output_dir) / robot_id / k).mkdir(parents=True, exist_ok=True)
    
    ### Make an initial guess for the extrinsic for each camera ###
    # CHECK: Double check this initial extrinsic guess is roughly close to the real world.
    initial_extrinsic_guesses = dict()
    for k in cameras_ft.keys():
        initial_extrinsic_guess = np.eye(4)

        # the guess says we are at position xyz=[-0.4, 0.0, 0.4] and angle the camerea downwards by np.pi / 4 radians  or 45 degrees
        # note that this convention is more natural for robotics (follows the typical convention for ROS and various simulators), where +Z is moving up towards the sky, +Y is to the left, +X is forward
        initial_extrinsic_guess[:3, :3] = euler2mat(np.pi/2, 0, -np.pi/3)
        # initial_extrinsic_guess[:3, 3] = np.array([0.3890, -0.4271, -2.3615])
        initial_extrinsic_guess[:3, 3] = np.array([-0.1, -0.3, 0.4]) # ver2
        # z->x, -y->z, x->z
        initial_extrinsic_guess[:3, 3] = np.array([0.24, 0.065, 0.63]) # ver2

        guess_transformation_matrix = np.array([
            [-6.4700e-01,  6.4018e-01, -4.1420e-01],
            [-5.3916e-01, -7.6822e-01, -3.4516e-01],
            [-5.3916e-01,  1.4901e-08,  8.4220e-01],
        ])
        print("transformation matrix in euler")
        r = R.from_matrix(guess_transformation_matrix)
        print(r.as_euler('xyz', degrees=True))

        initial_extrinsic_guess[:3, :3] = guess_transformation_matrix
        initial_extrinsic_guess[:3, :3] = euler2mat(5.32991492e-05,  3.26266428e+01, -1.40194676e+02)
        initial_extrinsic_guess[:3, :3] = euler2mat(-0.8,  39.95, -90.52)

        # pose:  tensor([[[-6.4700e-01,  6.4018e-01, -4.1420e-01, -1.0000e-01],
        #  [-5.3916e-01, -7.6822e-01, -3.4516e-01,  3.0000e-01],
        #  [-5.3916e-01,  1.4901e-08,  8.4220e-01,  4.0000e-01],
        #  [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]])
        # initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.2, 0.25])



        # initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)

        initial_extrinsic_guesses[k] = initial_extrinsic_guess

    print("Initial extrinsic guesses")
    for k in initial_extrinsic_guesses.keys():
        print(f"Camera {k}:\n{repr(initial_extrinsic_guesses[k])}")


    # get camera intrinsics for realsense cameras.
    intrinsics = dict()
    for cam_name, cam in robot.cameras.items():
        if isinstance(cam, RealSenseCamera):
            streams = cam.rs_profile.get_streams()
            assert len(streams) == 1, "Only one stream per camera is supported at the moment and it must be the color steam. Make sure to not enable any other streams."
            color_stream = streams[0]
            color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            intrinsic = np.array(
                [
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1],
                ]
            )
            intrinsics[cam_name] = intrinsic



    ### Data Collection Process below ###
    # We move the robot to a few joint configurations and collect images and generate a link pose dataset.

    joint_position_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    def get_qpos(robot: SO100Follower, flat: bool = True):
        obs = robot.bus.sync_read("Present_Position")
        for k in CALIBRATION_OFFSET.keys():
            obs[k] = obs[k] - CALIBRATION_OFFSET[k]
        for k in obs.keys():
            obs[k] = np.deg2rad(obs[k])
        if not flat:
            return obs
        joint_positions = []
        for k, v in obs.items():
            joint_positions.append(v)
        joint_positions = np.array(joint_positions)
        return joint_positions
    
    def set_target_qpos(robot: SO100Follower, qpos: np.ndarray):
        action = {}
        for name, qpos_val in zip(joint_position_names, qpos):
            action[name] = np.rad2deg(qpos_val) + CALIBRATION_OFFSET[name.removesuffix(".pos")]
        robot.send_action(action)
    
    robot_def_path = ROBOT_DEFINITIONS_DIR / "so100"
    robot_urdf = URDF.load(str(robot_def_path / "so100.urdf"))

    meshes = []
    for link in robot_urdf.links:
        link_meshes = []
        for visual in link.visuals:
            link_meshes += visual.geometry.mesh.meshes
        meshes.append(merge_meshes(link_meshes))

    if args.use_previous_captures and (Path(args.output_dir) / robot_id / "link_poses_dataset.npy").exists():
        # load the previous captures
        link_poses_dataset = np.load(Path(args.output_dir) / robot_id / "link_poses_dataset.npy")
        image_dataset = np.load(Path(args.output_dir) / robot_id / "image_dataset.npy", allow_pickle=True).reshape(-1)[0]
    else:
        # reference qpos positions to calibrate with    
        qpos_samples = [
            np.array([
                0, 0, 0, np.pi / 2, np.pi / 2, 0.2
            ]),
            # np.array([
            #     np.pi / 3, -np.pi / 6, 0, np.pi / 2, np.pi / 2, 0
            # ])
        ]
        control_freq = 15
        max_radians_per_step = 0.05

        # generate our link pose dataset and image pairs. We do this by moving the robot to the reference joint positions and collecting images from all cameras
        link_poses_dataset = np.zeros((len(qpos_samples), len(meshes), 4, 4))
        image_dataset = defaultdict(list)

        for i in range(len(qpos_samples)):

            # control code for lerobot below
            goal_qpos = qpos_samples[i]
            target_qpos = get_qpos(robot)
            for _ in range(int(20*control_freq)):
                start_loop_t = time.perf_counter()
                delta_qpos = (goal_qpos - target_qpos)
                delta_step = delta_qpos.clip(
                    min=-max_radians_per_step, max=max_radians_per_step
                )
                if np.linalg.norm(delta_qpos) < 1e-4:
                    break
                target_qpos += delta_step
                dt_s = time.perf_counter() - start_loop_t
                set_target_qpos(robot, target_qpos)
                time.sleep(1 / control_freq - dt_s)
            time.sleep(1) # give some time for the robot to settle, cheap arms don't hold up as well
            qpos_dict = get_qpos(robot, flat=False)
            for cam_name, cam in robot.cameras.items():
                image_dataset[cam_name].append(cam.async_read())
                
            # get link poses
            cfg = dict()
            for k in robot_urdf.joint_map.keys():
                cfg[k] = qpos_dict[k]
            link_poses = robot_urdf.link_fk(cfg=cfg, use_names=True)
            for link_idx, v in enumerate(link_poses.values()):
                link_poses_dataset[i, link_idx] = v
        for k in image_dataset.keys():
            image_dataset[k] = np.stack(image_dataset[k])

        np.save(Path(args.output_dir) / robot_id / "link_poses_dataset.npy", link_poses_dataset)
        np.save(Path(args.output_dir) / robot_id / "image_dataset.npy", image_dataset)

    ### Camera Calibration Process below ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in initial_extrinsic_guesses.keys():
        print(f"Calibrating camera {k}")
        initial_extrinsic_guess = initial_extrinsic_guesses[k]
        intrinsic = intrinsics[k]
        images = image_dataset[k]

        ### Print predicted results ###

        np.save(Path(args.output_dir) / robot_id / k / "camera_intrinsic.npy", intrinsic)

        visualization.visualize_extrinsic_results(
            images=images,
            link_poses_dataset=link_poses_dataset,
            meshes=meshes,
            intrinsic=intrinsic,
            extrinsics=np.stack(
                [initial_extrinsic_guess]
            ),
            masks=None,
            labels=["Initial Extrinsic Guess", "Predicted Extrinsic"],
            output_dir=str(Path(args.output_dir) / robot_id / k),
        )
        print(f"Visualizations saved to {Path(args.output_dir) / robot_id / k}")

if __name__ == "__main__":
    main(tyro.cli(SO100Args))