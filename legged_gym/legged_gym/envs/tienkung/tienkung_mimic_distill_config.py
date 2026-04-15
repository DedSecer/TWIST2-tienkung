from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class TienkungMimicPrivCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        # usually student obs_steps is the subset of priv obs_steps
        tar_motion_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 20  # Tienkung: Leg (2*6=12) + Arm (2*4=8) = 20
        obs_type = 'priv' # 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_motion_steps_priv) * (21 + num_actions + 3*9) # 9 key bodies
        n_mimic_obs_single = 6 + 20  # root_vel_xy(2) + root_pos_z(1) + roll_pitch(2) + yaw_ang_vel(1) + dof_pos(20)
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_priv_info = 3 + 3 + 4 + 3*9 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_priv_obs_single

        num_privileged_obs = n_priv_obs_single

        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = False
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
        
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.9
        rand_reset = True
        
        track_root = False
        root_tracking_termination_dist = 2.0
     
        # Tienkung: 12 leg DOFs + 8 arm DOFs = 20
        dof_err_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Left Leg (hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll)
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Right Leg
                     1.0, 1.0, 1.0, 1.0,             # Left Arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
                     1.0, 1.0, 1.0, 1.0,             # Right Arm
                     ]
        
        global_obs = False
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'plane'
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 1.0]  # Tienkung standing height (from MJCF: pelvis pos="0 0 1.0")
        default_joint_angles = {
            # Left Leg
            'hip_roll_l_joint':   0.0,
            'hip_pitch_l_joint':  -0.5,
            'hip_yaw_l_joint':    0.0,
            'knee_pitch_l_joint': 1.0,
            'ankle_pitch_l_joint': -0.5,
            'ankle_roll_l_joint':  0.0,

            # Right Leg
            'hip_roll_r_joint':   0.0,
            'hip_pitch_r_joint':  -0.5,
            'hip_yaw_r_joint':    0.0,
            'knee_pitch_r_joint': 1.0,
            'ankle_pitch_r_joint': -0.5,
            'ankle_roll_r_joint':  0.0,

            # Left Arm — closer to body
            'shoulder_pitch_l_joint': 0.0,
            'shoulder_roll_l_joint':  0.1,
            'shoulder_yaw_l_joint':   0.0,
            'elbow_pitch_l_joint':    -0.3,

            # Right Arm — closer to body
            'shoulder_pitch_r_joint': 0.0,
            'shoulder_roll_r_joint':  -0.1,
            'shoulder_yaw_r_joint':   0.0,
            'elbow_pitch_r_joint':    -0.3,
        }
    
    class control(HumanoidMimicCfg.control):
        # PD gains sourced directly from GMR/assets/tienkung/mjcf/tienkung.xml
        # stiffness = kp values from <actuator> <position kp=...> tags
        # damping   = damping values from <joint damping=...> tags
        stiffness = {
            'hip_roll':  700,
            'hip_pitch': 700,
            'hip_yaw':   500,
            'knee':      700,
            'ankle_pitch': 30,
            'ankle_roll':  16.8,
            'shoulder_pitch': 60,
            'shoulder_roll':  20,
            'shoulder_yaw':   10,
            'elbow':          10,
        }  # [N*m/rad]
        damping = {
            # Legs - from MJCF joint damping attributes
            'hip_roll':  10,
            'hip_pitch': 10,
            'hip_yaw':   5,
            'knee':      10,
            # Ankles
            'ankle_pitch': 2.5,
            'ankle_roll':  1.4,
            # Arms
            'shoulder_pitch': 3,
            'shoulder_roll':  1.5,
            'shoulder_yaw':   1,
            'elbow':          1,
        }  # [N*m*s/rad]

        action_scale = 0.5
        decimation = 10
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002 # 1/500
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/tienkung/urdf/tienkung2_lite.urdf'
        
        # for both joint and link name
        torso_name: str = 'pelvis'  # humanoid pelvis part (root link in URDF)
        chest_name: str = 'pelvis'  # Tienkung doesn't have separate chest, use pelvis

        # for link name
        thigh_name: str = 'hip'
        shank_name: str = 'knee'
        foot_name: str = 'ankle_roll'  
        waist_name: list = []  # Tienkung has no waist joints
        upper_arm_name: str = 'shoulder_roll'
        lower_arm_name: str = 'elbow'
        hand_name: list = ['left_hand', 'right_hand']

        feet_bodies = ['ankle_roll_l_link', 'ankle_roll_r_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = []
        
        # Inertia values - estimated for Tienkung motors
        # Leg motors (hip, knee): larger actuators
        # Arm motors (shoulder, elbow): smaller actuators
        dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + \
            [0.003597, 0.003597, 0.003597, 0.003597] * 2
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = []
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        
        class scales:      
            tracking_joint_dof = 2.0
            tracking_joint_vel = 0.2
            tracking_root_translation_z = 1.0
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.0
            tracking_root_angular_vel = 1.0
            tracking_keybody_pos = 2.0
            tracking_keybody_pos_global = 2.0
            alive = 0.5
            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            feet_stumble = -1.25
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            feet_air_time = 5.0
            ang_vel_xy = -0.01            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        
        # =========================
        termination_roll = 4.0
        termination_pitch = 4.0
        root_height_diff_threshold = 0.3
        

    class evaluations:
        tracking_joint_dof = True
        tracking_joint_vel = True
        tracking_root_translation = True
        tracking_root_rotation = True
        tracking_root_vel = True
        tracking_root_ang_vel = True
        tracking_keybody_pos = True
        tracking_root_pose_delta_local = True
        tracking_root_rotation_delta_local = True
        
        
    class domain_rand:
        domain_rand_general = True # manually set this, setting from parser does not work;
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (False and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 10.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 50_000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
        
    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        reset_consec_frames = 30
        # Tienkung key bodies: hands, feet, knees, elbows, head = 9 key bodies
        key_bodies = ["left_hand", "right_hand", "ankle_roll_l_link", "ankle_roll_r_link", "knee_pitch_l_link", "knee_pitch_r_link", "elbow_pitch_l_link", "elbow_pitch_r_link", "head"]
        upper_key_bodies = ["left_hand", "right_hand", "elbow_pitch_l_link", "elbow_pitch_r_link", "head"]
        sample_ratio = 1.0
        motion_smooth = True
        motion_decompose = False

        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/tienkung_dataset.yaml"
        


class TienkungMimicStuCfg(TienkungMimicPrivCfg):
    class env(TienkungMimicPrivCfg.env):
        obs_type = 'student'
        tar_motion_steps = [1]
        n_mimic_obs_single = TienkungMimicPrivCfg.env.n_mimic_obs_single
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = TienkungMimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (TienkungMimicPrivCfg.env.history_len + 1)


class TienkungMimicStuRLCfg(TienkungMimicPrivCfg):
    class env(TienkungMimicPrivCfg.env):
        obs_type = 'student'
        tar_motion_steps = [1]
        n_mimic_obs_single = TienkungMimicPrivCfg.env.n_mimic_obs_single
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = TienkungMimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (TienkungMimicPrivCfg.env.history_len + 1)



class TienkungMimicPrivCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 1_000_002 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
    
    class policy(HumanoidMimicCfgPPO.policy):
        # Tienkung: 12 leg + 8 arm = 20 DOFs
        action_std = [0.7] * 12 + [0.5] * 8
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        
        
class TienkungMimicStuCfgDAgger(TienkungMimicPrivCfgPPO):
    seed = 1
    
    class teachercfg(TienkungMimicPrivCfgPPO):
        pass
    
    class runner(TienkungMimicPrivCfgPPO.runner):
        policy_class_name = 'DAggerActor'
        algorithm_class_name = 'DAgger'
        runner_class_name = 'DAggerRunner'
        max_iterations = 1_000_002
        warm_iters = 100
        
        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'tienkung_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm:
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-4
        max_grad_norm = 1.0
        normalizer_update_iterations = 1000

    class policy:
        actor_hidden_dims = [1024, 1024, 512, 256]
        history_latent_dim = 128
        activation = 'silu'
        

class TienkungMimicStuRLCfgDAgger(TienkungMimicStuRLCfg):
    seed = 1
    
    class teachercfg(TienkungMimicPrivCfgPPO):
        pass
    
    class runner(TienkungMimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticTeleop'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 1_000_002
        warm_iters = 100
        
        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'tienkung_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.5] * 8
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
