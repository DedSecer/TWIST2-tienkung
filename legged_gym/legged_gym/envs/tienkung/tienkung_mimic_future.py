"""Tienkung mimic future environment.
Inherits from G1MimicFuture - the core environment logic is robot-agnostic,
all robot-specific settings come from the config files.
"""
from legged_gym.envs.g1.g1_mimic_future import G1MimicFuture


class TienkungMimicFuture(G1MimicFuture):
    """Tienkung humanoid robot environment with future motion support.
    
    This class inherits all functionality from G1MimicFuture.
    Robot-specific parameters (DOFs, joint names, URDF, etc.) are configured
    in tienkung_mimic_future_config.py.
    """
    pass
