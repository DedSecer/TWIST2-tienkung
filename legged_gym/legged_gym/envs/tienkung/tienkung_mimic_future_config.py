from legged_gym.envs.tienkung.tienkung_mimic_distill_config import TienkungMimicPrivCfg, TienkungMimicPrivCfgPPO
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


TAR_MOTION_STEPS_FUTURE = [0]
class TienkungMimicStuFutureCfg(TienkungMimicPrivCfg):
    """Student policy config with future motion support and curriculum masking.
    Extends existing TienkungMimicPrivCfg to add future motion capabilities."""
    
    class env(TienkungMimicPrivCfg.env):
        obs_type = 'student_future'
        
        # Keep original student motion steps (current frame only)
        tar_motion_steps = [0]
        
        # Future motion frames
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE
        
        # Observation dimensions (Tienkung: 20 DOFs)
        n_mimic_obs_single = 6 + 20
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single  # Current frame only
        n_proprio = TienkungMimicPrivCfg.env.n_proprio

        # Future observation dimensions
        n_future_obs_single = 6 + 20  # Masking disabled -> no indicator channel
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single
        
        # Total observation size: maintain original structure + future observations
        n_obs_single = n_mimic_obs + n_proprio  # Current frame observation (for history)
        num_observations = n_obs_single * (TienkungMimicPrivCfg.env.history_len + 1) + n_future_obs
        
        
        # FALCON-style curriculum force application (domain randomization)
        enable_force_curriculum = False
        
        class force_curriculum:
            # Force application settings - use Tienkung hand names
            force_apply_links = ['left_hand', 'right_hand']
            
            # Force curriculum learning
            force_scale_curriculum = True
            force_scale_initial_scale = 1.0
            force_scale_up_threshold = 210
            force_scale_down_threshold = 200
            force_scale_up = 0.02
            force_scale_down = 0.02
            force_scale_max = 1.0
            force_scale_min = 0.0
            
            # Force application ranges (in Newtons)
            apply_force_x_range = [-40.0, 40.0]
            apply_force_y_range = [-40.0, 40.0]
            apply_force_z_range = [-50.0, 5.0]
            
            # Force randomization
            zero_force_prob = [0.25, 0.25, 0.25]
            randomize_force_duration = [10, 50]
            
            # Advanced force settings
            max_force_estimation = True
            use_lpf = False
            force_filter_alpha = 0.05
            
            # Task-specific force behavior
            only_apply_z_force_when_walking = False
            only_apply_resistance_when_walking = True
    
    class motion(TienkungMimicPrivCfg.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/tienkung_dataset.yaml"
        
        # Ensure motion curriculum is enabled for difficulty adaptation
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        motion_decompose = False

        # Motion Domain Randomization
        motion_dr_enabled = False
        root_position_noise = [0.01, 0.05]
        root_orientation_noise = [0.1, 0.2]
        root_velocity_noise = [0.05, 0.1]
        joint_position_noise = [0.05, 0.1]
        motion_dr_resampling = True
        
        # Error Aware Sampling parameters
        use_error_aware_sampling = False
        error_sampling_power = 5.0
        error_sampling_threshold = 0.15
    
    class rewards(TienkungMimicPrivCfg.rewards):
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
            action_rate = -0.05
            feet_air_time = 5.0
            ang_vel_xy = -0.01            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
        


class TienkungMimicStuFutureCfgDAgger(TienkungMimicStuFutureCfg):
    """DAgger training config for future motion student policy."""
    
    seed = 1
    
    class teachercfg(TienkungMimicPrivCfgPPO):
        pass
    
    class runner(TienkungMimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 30_001
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
        
        # Wandb model saving option
        save_to_wandb = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        
        # Future motion specific parameters
        future_weight_decay = 0.95
        future_consistency_loss = 0.1

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
        
        # Future motion encoder parameters
        future_encoder_dims = [256, 256, 128]
        future_attention_heads = 4
        future_dropout = 0.1
        temporal_embedding_dim = 64
        future_latent_dim = 128
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE)
        
        # Explicit future observation dimensions
        num_future_observations = TienkungMimicStuFutureCfg.env.n_future_obs
        
        # MoE specific parameters
        num_experts = 4
        expert_hidden_dims = [256, 128]
        gating_hidden_dim = 128
        moe_temperature = 1.0
        moe_topk = None
        load_balancing_loss_weight = 0.01
