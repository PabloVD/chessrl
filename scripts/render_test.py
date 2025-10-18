import chess
import matplotlib.pyplot as plt

from chessrl import ChessRenderer

board = chess.Board()
renderer = ChessRenderer(square_size=80)

for move in ["e4", "e5", "Nf3", "Nc6"]:
    board.push_san(move)
    renderer.render(board)

# Keep the window open at the end
plt.ioff()
plt.show()
