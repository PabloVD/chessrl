from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from chessrl import ChessEnv, SelfPlayWrapper

checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path="./checkpoints/",
    name_prefix="chess_agent"
)

env = SelfPlayWrapper(ChessEnv())

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=256,
    batch_size=256,
    tensorboard_log="./chess_tensorboard/"
)

model.learn(total_timesteps=100_000, callback=checkpoint_callback)
