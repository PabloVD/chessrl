import gymnasium as gym
from gymnasium import spaces
import numpy as np
import chess
from .chessrenderer import ChessRenderer

class ChessEnv(gym.Env):
    def __init__(self, render_mode: str = "human"):
        super().__init__()
        self.board : chess.Board = chess.Board()
        self.action_space = spaces.Discrete(4672)  # max legal moves in chess
        self.observation_space = spaces.Box(0, 1, (12, 8, 8), dtype=np.uint8)
        self.render_mode = render_mode
        self.renderer = ChessRenderer()
    
    def _board_to_obs(self):
        obs = np.zeros((12, 8, 8), dtype=np.uint8)  # note: channel-first + uint8
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                c = piece.piece_type - 1 + (0 if piece.color else 6)
                obs[c, i // 8, i % 8] = 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self._board_to_obs(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            return self._board_to_obs(), 0.0, True, False, {}

        if action >= len(legal_moves):
            # Invalid move
            return self._board_to_obs(), -1.0, True, False, {"invalid_action": True}

        move = legal_moves[action]
        self.board.push(move)
        reward = self._get_reward()
        done = self.board.is_game_over()
        return self._board_to_obs(), reward, done, False, {}

    def _get_reward(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        return 0
    
    def render(self):
        """Render the board based on the selected mode."""
        if self.render_mode == "human":
            self.renderer.render(self.board)
        else:
            print(self.board)


