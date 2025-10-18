from stable_baselines3 import PPO
from chessrl import ChessEnv, SelfPlayWrapper

env = SelfPlayWrapper(ChessEnv())

# from stable_baselines3.common.env_checker import check_env
# check_env(env)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=256,
    batch_size=256,
    tensorboard_log="./chess_tensorboard/"
)

model.learn(total_timesteps=1_000_000)
