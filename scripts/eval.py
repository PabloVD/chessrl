from stable_baselines3 import PPO
from chessrl import ChessEnv, SelfPlayWrapper
import time

env = SelfPlayWrapper(ChessEnv(render_mode="human"))

model = PPO.load("checkpoints/chess_agent_800000_steps.zip", env=env)

obs, _ = env.reset()
done = False
step_count = 0

while not done and step_count < 200:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    time.sleep(0.5)
    step_count += 1

print("Game over:", env.env.board.result())
