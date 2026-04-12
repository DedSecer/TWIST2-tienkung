import argparse
import json
import time
import numpy as np
import redis
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os
from data_utils.rot_utils import quatToEuler

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


class TienkungSimController:
    """Sim2sim controller for Tienkung humanoid robot (20 DOFs)."""

    def __init__(self, 
                 xml_file, 
                 policy_path, 
                 device='cuda', 
                 record_video=False,
                 record_proprio=False,
                 measure_fps=False,
                 limit_fps=True,
                 policy_frequency=50,
                 ):
        self.measure_fps = measure_fps
        self.limit_fps = limit_fps
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.cam.distance = 2.0

        # Tienkung has 20 DOFs: 12 leg + 8 arm
        self.num_actions = 20
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        # real frequency = 1 / (decimation * sim_dt)
        # ==> decimation = 1 / (real frequency * sim_dt)
        self.sim_decimation = 1 / (policy_frequency * self.sim_dt)
        print(f"sim_decimation: {self.sim_decimation}")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # Tienkung specific configuration
        # DOF order: hip_roll_l, hip_pitch_l, hip_yaw_l, knee_pitch_l, ankle_pitch_l, ankle_roll_l,
        #            hip_roll_r, hip_pitch_r, hip_yaw_r, knee_pitch_r, ankle_pitch_r, ankle_roll_r,
        #            shoulder_pitch_l, shoulder_roll_l, shoulder_yaw_l, elbow_pitch_l,
        #            shoulder_pitch_r, shoulder_roll_r, shoulder_yaw_r, elbow_pitch_r
        # Default DOF positions — from official TienKung-Lab reference config
        self.default_dof_pos = np.array([
                0.0, -0.5, 0.0, 1.0, -0.5, 0.0,   # left leg:  hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll
                0.0, -0.5, 0.0, 1.0, -0.5, 0.0,   # right leg
                0.0,  0.1, 0.0, -0.3,              # left arm:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
                0.0, -0.1, 0.0, -0.3,              # right arm
            ])

        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 1.0]),      # root position (x, y, z) — from MJCF Base_link pos
            np.array([1, 0, 0, 0]),     # root quaternion (w, x, y, z)
            np.array([0.0, -0.5, 0.0, 1.0, -0.5, 0.0,   # left leg
                      0.0, -0.5, 0.0, 1.0, -0.5, 0.0,   # right leg
                      0.0,  0.1, 0.0, -0.3,              # left arm
                      0.0, -0.1, 0.0, -0.3,              # right arm
                ])
        ])

        # PD gains — from official TienKung-Lab reference config (same source as MJCF kp/damping)
        self.stiffness = np.array([
                700, 700, 500, 700, 30,   16.8,   # left leg:  hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll
                700, 700, 500, 700, 30,   16.8,   # right leg
                60,  20,  10,  10,                # left arm:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
                60,  20,  10,  10,                # right arm
            ])
        self.damping = np.array([
                10,  10,  5,   10,  2.5, 1.4,    # left leg
                10,  10,  5,   10,  2.5, 1.4,    # right leg
                3,   1.5, 1,   1,                 # left arm
                3,   1.5, 1,   1,                 # right arm
            ])

        self.torque_limits = np.array([
                180, 300, 180, 300, 60,  30,      # left leg  — from effort_limit_sim in TienKung-Lab cfg
                180, 300, 180, 300, 60,  30,      # right leg
                52.5, 52.5, 52.5, 52.5,           # left arm
                52.5, 52.5, 52.5, 52.5,           # right arm
            ])

        self.action_scale = np.array([
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # left leg
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # right leg
                0.5, 0.5, 0.5, 0.5,             # left arm
                0.5, 0.5, 0.5, 0.5,             # right arm
            ])

        # Ankle indices (for zeroing velocity in obs)
        self.ankle_idx = [4, 5, 10, 11]

        # Observation structure for Tienkung (20 DOFs)
        # n_mimic_obs_single = 6 + 20 = 26
        # n_proprio = 3 + 2 + 3*20 = 65
        # n_obs_single = 26 + 65 = 91
        # total_obs = n_obs_single * (history_len + 1) + n_future_obs
        #           = 91 * 11 + 26 = 1001 + 26 = 1027
        self.n_mimic_obs = 26  # 6 + 20 (root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos)
        self.n_proprio = 3 + 2 + 3*20    # ang_vel*0.25(3) + roll_pitch(2) + dof_pos(20) + dof_vel(20) + last_action(20)
        self.n_obs_single = 26 + 3 + 2 + 3*20  # n_mimic_obs + n_proprio = 26 + 65 = 91
        self.history_len = 10
        
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs   # 91*11 + 26 = 1027

        print(f"Tienkung Sim2Sim Controller Configuration:")
        print(f"  num_actions: {self.num_actions}")
        print(f"  n_mimic_obs: {self.n_mimic_obs}")
        print(f"  n_proprio: {self.n_proprio}")
        print(f"  n_obs_single: {self.n_obs_single}")
        print(f"  history_len: {self.history_len}")
        print(f"  total_obs_size: {self.total_obs_size}")

        # Initialize history buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

        # Recording
        self.record_video = record_video
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None
        

    def reset_sim(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, init_pos):
        """Reset robot to initial position"""
        self.data.qpos[:] = init_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        """Extract robot state data"""
        n_dof = self.num_actions
        dof_pos = self.data.qpos[7:7+n_dof]
        dof_vel = self.data.qvel[6:6+n_dof]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        sim_torque = self.data.ctrl
        return dof_pos, dof_vel, quat, ang_vel, sim_torque

    def run(self):
        """Main simulation loop"""
        print("Starting Tienkung sim2sim simulation...")

        # Video recording setup
        if self.record_video:
            import imageio
            mp4_writer = imageio.get_writer('tienkung_sim2sim.mp4', fps=30)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating Tienkung...")

        # Send initial proprio to redis
        initial_obs = np.zeros(self.n_obs_single, dtype=np.float32)
        self.redis_pipeline.set("state_body_tienkung", json.dumps(initial_obs.tolist()))
        self.redis_pipeline.set("state_hand_left_tienkung", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("state_hand_right_tienkung", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.execute()

        measure_fps = self.measure_fps
        fps_measurements = []
        fps_iteration_count = 0
        fps_measurement_target = 1000
        last_policy_time = None

        # Policy execution FPS tracking
        policy_execution_times = []
        policy_step_count = 0
        policy_fps_print_interval = 100

        try:
            for i in pbar:
                t_start = time.time()
                dof_pos, dof_vel, quat, ang_vel, sim_torque = self.extract_data()
                
                if i % self.sim_decimation == 0:
                    # Build proprioceptive observation
                    rpy = quatToEuler(quat)
                    obs_body_dof_vel = dof_vel.copy()
                    obs_body_dof_vel[self.ankle_idx] = 0.
                    obs_proprio = np.concatenate([
                        ang_vel * 0.25,
                        rpy[:2], # only use roll and pitch
                        (dof_pos - self.default_dof_pos),
                        obs_body_dof_vel * 0.05,
                        self.last_action
                    ])

                    state_body = np.concatenate([
                        ang_vel,
                        rpy[:2],
                        dof_pos]) # 3+2+20 = 25 dims

                    # Send proprio to redis
                    self.redis_pipeline.set("state_body_tienkung", json.dumps(state_body.tolist()))
                    self.redis_pipeline.set("state_hand_left_tienkung", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_hand_right_tienkung", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_neck_tienkung", json.dumps(np.zeros(2).tolist()))
                    self.redis_pipeline.set("t_state", int(time.time() * 1000)) # current timestamp in ms
                    self.redis_pipeline.execute()

                    # Get mimic obs from Redis
                    keys = ["action_body_tienkung", "action_hand_left_tienkung", "action_hand_right_tienkung", "action_neck_tienkung"]
                    for key in keys:
                        self.redis_pipeline.get(key)
                    redis_results = self.redis_pipeline.execute()
                    action_mimic = json.loads(redis_results[0])
                    action_left_hand = json.loads(redis_results[1])
                    action_right_hand = json.loads(redis_results[2])
                    action_neck = json.loads(redis_results[3])

                    # Construct observation for Tienkung controller
                    obs_full = np.concatenate([action_mimic, obs_proprio])
                    # Update history
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    self.proprio_history_buf.append(obs_full)
                    future_obs = action_mimic.copy()
                    # Combine all observations: current + history + future
                    obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
                    

                    # Ensure correct total observation size
                    assert obs_buf.shape[0] == self.total_obs_size, f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
                    
                    # Run policy
                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                    
                    # Measure and track policy execution FPS
                    current_time = time.time()
                    if last_policy_time is not None:
                        policy_interval = current_time - last_policy_time
                        current_policy_fps = 1.0 / policy_interval
                        
                        policy_execution_times.append(policy_interval)
                        policy_step_count += 1
                        
                        # Print policy execution FPS every 100 steps
                        if policy_step_count % policy_fps_print_interval == 0:
                            recent_intervals = policy_execution_times[-policy_fps_print_interval:]
                            avg_interval = np.mean(recent_intervals)
                            avg_execution_fps = 1.0 / avg_interval
                            print(f"Policy Execution FPS (last {policy_fps_print_interval} steps): {avg_execution_fps:.2f} Hz (avg interval: {avg_interval*1000:.2f}ms)")
                        
                        if measure_fps:
                            fps_measurements.append(current_policy_fps)
                            fps_iteration_count += 1
                            
                            if fps_iteration_count == fps_measurement_target:
                                avg_fps = np.mean(fps_measurements)
                                max_fps = np.max(fps_measurements)
                                min_fps = np.min(fps_measurements)
                                std_fps = np.std(fps_measurements)
                                print(f"\n=== Policy Execution FPS Results (steps {fps_iteration_count-fps_measurement_target+1}-{fps_iteration_count}) ===")
                                print(f"Average Policy FPS: {avg_fps:.2f}")
                                print(f"Max Policy FPS: {max_fps:.2f}")
                                print(f"Min Policy FPS: {min_fps:.2f}")
                                print(f"Std Policy FPS: {std_fps:.2f}")
                                print(f"Expected FPS (from decimation): {1.0/(self.sim_decimation * self.sim_dt):.2f}")
                                print(f"=================================================================================\n")
                                fps_measurements = []
                                fps_iteration_count = 0
                    last_policy_time = current_time
                    
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos

                    # Update camera to follow pelvis (Base_link in Tienkung)
                    pelvis_pos = self.data.xpos[self.model.body("Base_link").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()
                    
                    if mp4_writer is not None:
                        img = self.viewer.read_pixels()
                        mp4_writer.append_data(img)

                    # Record proprio if enabled
                    if self.record_proprio:
                        proprio_data = {
                            'timestamp': time.time(),
                            'dof_pos': dof_pos.tolist(),
                            'dof_vel': dof_vel.tolist(),
                            'rpy': rpy.tolist(),
                            'ang_vel': ang_vel.tolist(),
                            'target_dof_pos': action_mimic[-20:],
                        }
                        self.proprio_recordings.append(proprio_data)

               
                # PD control
                torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                
                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)
                
                # Sleep to maintain real-time pace
                if self.limit_fps:
                    elapsed = time.time() - t_start
                    if elapsed < self.sim_dt:
                        time.sleep(self.sim_dt - elapsed)

                    
        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved as tienkung_sim2sim.mp4")
            
            # Save proprio recordings if enabled
            if self.record_proprio and self.proprio_recordings:
                import pickle
                with open('tienkung_proprio_recordings.pkl', 'wb') as f:
                    pickle.dump(self.proprio_recordings, f)
                print("Proprioceptive recordings saved as tienkung_proprio_recordings.pkl")

            if self.viewer:
                self.viewer.close()
            print("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description='Run Tienkung policy in sim2sim simulation')
    parser.add_argument('--xml', type=str, default='../assets/tienkung/mjcf/tienkung.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to Tienkung ONNX policy file')
    parser.add_argument('--device', type=str, 
                        default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--record_video', action='store_true',
                        help='Record video of simulation')
    parser.add_argument('--record_proprio', action='store_true',
                        help='Record proprioceptive data')
    parser.add_argument("--measure_fps", help="Measure FPS", default=0, type=int)
    parser.add_argument("--limit_fps", help="Limit FPS with sleep", default=1, type=int)
    parser.add_argument("--policy_frequency", help="Policy frequency", default=50, type=int)
    args = parser.parse_args()
    
    # Verify policy file exists
    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
    
    # Verify XML file exists
    if not os.path.exists(args.xml):
        print(f"Error: XML file {args.xml} does not exist")
        return
    
    print(f"Starting Tienkung sim2sim controller...")
    print(f"  XML file: {args.xml}")
    print(f"  Policy file: {args.policy}")
    print(f"  Device: {args.device}")
    print(f"  Record video: {args.record_video}")
    print(f"  Record proprio: {args.record_proprio}")
    print(f"  Measure FPS: {args.measure_fps}")
    print(f"  Limit FPS: {args.limit_fps}")
    controller = TienkungSimController(
        xml_file=args.xml,
        policy_path=args.policy,
        device=args.device,
        record_video=args.record_video,
        record_proprio=args.record_proprio,
        measure_fps=args.measure_fps,
        limit_fps=args.limit_fps,
        policy_frequency=args.policy_frequency,
    )
    controller.run()


if __name__ == "__main__":
    main()
