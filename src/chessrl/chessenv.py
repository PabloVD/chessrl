import gymnasium as gym
from gymnasium import spaces
import numpy as np
import chess

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board : chess.Board = chess.Board()
        self.action_space = spaces.Discrete(4672)  # max legal moves in chess
        self.observation_space = spaces.Box(0, 1, (8, 8, 12), dtype=np.int8)

    def _board_to_obs(self):
        obs = np.zeros((8, 8, 12), dtype=np.int8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                obs[i // 8, i % 8, piece.piece_type - 1 + (0 if piece.color else 6)] = 1
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self._board_to_obs(), {}

    def step(self, action):
        move = list(self.board.legal_moves)[action]
        self.board.push(move)
        reward = self._get_reward()
        done = self.board.is_game_over()
        return self._board_to_obs(), reward, done, False, {}

    def _get_reward(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        return 0

