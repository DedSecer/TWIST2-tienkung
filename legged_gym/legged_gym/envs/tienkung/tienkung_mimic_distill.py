"""Tienkung mimic distill environment.
Inherits from G1MimicDistill - the core environment logic is robot-agnostic,
all robot-specific settings come from the config files.
"""
from legged_gym.envs.g1.g1_mimic_distill import G1MimicDistill


class TienkungMimicDistill(G1MimicDistill):
    """Tienkung humanoid robot environment for motion imitation with distillation.
    
    This class inherits all functionality from G1MimicDistill.
    Robot-specific parameters (DOFs, joint names, URDF, etc.) are configured
    in tienkung_mimic_distill_config.py.
    """
    pass
