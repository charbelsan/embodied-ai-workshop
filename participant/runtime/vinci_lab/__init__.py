"""VINCI Robotics Lab: one robot, one scene, several ways to build a skill, one Brain that composes them."""
try:  # the notebook client (vinci_lab.client) must import without the Isaac stack
    import gymnasium as gym
except ImportError:
    gym = None

if gym is not None:
  gym.register(
    id="Vinci-Open-Drawer-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "vinci_lab.scene:VinciOpenDrawerRLEnvCfg",
        "rsl_rl_cfg_entry_point": "vinci_lab.scene:VinciOpenDrawerPPORunnerCfg",
    },
)
  gym.register(
    id="Vinci-Open-Drawer-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "vinci_lab.scene:VinciOpenDrawerRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "vinci_lab.scene:VinciOpenDrawerPPORunnerCfg",
    },
)
