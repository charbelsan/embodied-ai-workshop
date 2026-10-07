"""Privileged teacher -> BC -> DAgger track (open_drawer). Importing registers the gym tasks (cfgs load lazily)."""
try:  # the student model (vinci_lab.dagger.student) must import without the Isaac stack
    import gymnasium as gym
except ImportError:
    gym = None

if gym is not None:
    gym.register(
        id="Vinci-Dagger-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "vinci_lab.dagger.env_cfg:VinciDaggerTeacherEnvCfg",
            "rsl_rl_cfg_entry_point": "vinci_lab.dagger.env_cfg:VinciDaggerTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Vinci-Dagger-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "vinci_lab.dagger.env_cfg:VinciDaggerStudentEnvCfg"},
    )
