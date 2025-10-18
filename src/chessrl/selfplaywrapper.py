import gymnasium as gym
import numpy as np
from .chessenv import ChessEnv

class SelfPlayWrapper(gym.Wrapper):
    """
    A wrapper for self-play training in symmetric games like chess.
    One policy plays both sides (White and Black) alternately.
    """

    def __init__(self, env: ChessEnv, mirror_obs=True):
        super().__init__(env)
        self.mirror_obs = mirror_obs
        self.current_player = True  # True = White, False = Black

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.current_player = True  # White starts
        if self.mirror_obs and not self.env.board.turn:
            obs = self._mirror_observation(obs)
        return obs, info

    def step(self, action):
        """
        Apply one move for the current player, then switch turns.
        The reward is always from the perspective of the player who just moved.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Reward should be from the current player's perspective
        # If black just moved, flip reward sign
        if not self.current_player:
            reward = -reward

        done = terminated or truncated

        if not done:
            # Switch turns: the same policy acts again for the opponent
            self.current_player = not self.current_player

            # Mirror board so the agent always sees itself as "white"
            if self.mirror_obs and not self.env.board.turn:
                obs = self._mirror_observation(obs)

        return obs, reward, done, truncated, info

    def _mirror_observation(self, obs):
        """
        Flip the observation along both axes and swap piece channels
        so black's perspective looks like white's.
        """
        # obs shape: (8, 8, 12)
        flipped = np.flip(obs, axis=(0, 1))
        # swap first 6 (white pieces) with last 6 (black pieces)
        swapped = np.concatenate([flipped[:, :, 6:], flipped[:, :, :6]], axis=-1)
        return swapped
