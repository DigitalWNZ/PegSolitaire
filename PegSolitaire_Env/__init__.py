from gymnasium.envs import register

register(
    id='pegsolitaire-v0',
    entry_point='PegSolitaire_Env.envs:PegSolitaireEnv',
    max_episode_steps=31,
    kwargs={"reward_range": (-100,100)},
)


