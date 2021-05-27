from gym.envs.registration import register

register(
    id='Ds3gym-v0',
    entry_point='gym_ds3.envs.core.ds3_env:DS3GymEnv',
)
