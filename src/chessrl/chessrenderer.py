import chess
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class ChessRenderer:
    def __init__(self, square_size=60):
        self.square_size = square_size
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.axis("off")
        plt.ion()  # interactive mode on
        self.img_obj = None

        # Load font
        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/seguisym.ttf", int(square_size*0.8))
        except IOError:
            self.font = ImageFont.load_default()
            print("Using default font; chess symbols may not render correctly.")

    def render(self, board: chess.Board):
        # Create image
        board_size = self.square_size * 8
        img = Image.new("RGB", (board_size, board_size), "white")
        draw = ImageDraw.Draw(img)

        light = (240, 217, 181)
        dark = (181, 136, 99)

        for rank in range(8):
            for file in range(8):
                color = dark if (rank + file) % 2 == 0 else light
                x0 = file * self.square_size
                y0 = (7 - rank) * self.square_size
                draw.rectangle([x0, y0, x0 + self.square_size, y0 + self.square_size], fill=color)

        piece_symbols = {
            "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
            "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"
        }

        for square, piece in board.piece_map().items():
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            x = file * self.square_size + self.square_size/8
            y = (7 - rank) * self.square_size + self.square_size/12
            draw.text((x, y), piece_symbols[str(piece)], fill="black", font=self.font)

        # Convert to array
        img_array = np.array(img)

        # Update the figure
        if self.img_obj is None:
            self.img_obj = self.ax.imshow(img_array)
        else:
            self.img_obj.set_data(img_array)

        plt.pause(0.5)  # control speed
        plt.draw()

