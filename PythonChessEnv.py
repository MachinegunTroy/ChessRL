import chess
import numpy as np
import gym
from gym import spaces
import tkinter as tk
from tkinter import Canvas
import time
import random
from PythonChessAgent import DQNNet, DQNAgent

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()  # Creates a new board.
        self.observation_space = spaces.Box(low=-6, high=6, shape=(8, 8), dtype=np.int8)
        # New action space: 20480 = 4096 base moves * 5 promotion options.
        self.action_space = spaces.Discrete(20480)

    def reset(self):
        self.board = chess.Board()
        return self.get_observation() 

    def get_observation(self):
        board_array = np.zeros((8, 8), dtype=np.int8)
        piece_to_int = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                value = piece_to_int[piece.piece_type]
                if piece.color == chess.BLACK:
                    value = -value
                board_array[row, col] = value
        return board_array

    def get_actions(self):
        return list(self.board.legal_moves)

    def step(self, action):
        # Store the color of the moving agent.
        moving_color = self.board.turn  # True for White, False for Black.

        # Decode promotion flag and the underlying move.
        promotion_flag = action // 4096  # 0 to 4.
        base_move = action % 4096         # 0 to 4095.
        from_square = base_move // 64
        to_square = base_move % 64

        # Get the moving piece.
        piece = self.board.piece_at(from_square)

        # Construct the move. For pawn moves reaching the last rank, use the promotion flag.
        move = chess.Move(from_square, to_square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
                promotion_mapping = {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}
                promotion_piece = promotion_mapping.get(promotion_flag, chess.QUEEN)
                move = chess.Move(from_square, to_square, promotion=promotion_piece)

        illegal = False
        # Check if the chosen move is illegal.
        if move not in self.board.legal_moves:
            illegal = True
            # Set reward for illegal move.
            reward = -1.0
            # Substitute a random legal move so that the game can continue.
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
            else:
                return self.get_observation(), reward, True, {"illegal_move": True}
        else:
            # For legal moves, start with a base reward of +1.
            reward = 1.0

        # Only add bonuses for legal moves (if not already penalized).
        if not illegal:
            # Additional reward for castling.
            if self.board.is_castling(move):
                reward += 0.5

            # Reward for capturing an opponent's piece.
            if self.board.is_capture(move):
                if self.board.is_en_passant(move):
                    captured_square = move.to_square - 8 if moving_color == chess.WHITE else move.to_square + 8
                else:
                    captured_square = move.to_square
                captured_piece = self.board.piece_at(captured_square)
                if captured_piece:
                    piece_values = {
                        chess.PAWN: 1,
                        chess.KNIGHT: 3,
                        chess.BISHOP: 3,
                        chess.ROOK: 5,
                        chess.QUEEN: 9,
                        chess.KING: 0
                    }
                    reward += piece_values[captured_piece.piece_type]

        # Execute the move.
        self.board.push(move)

        # Check for game termination.
        done = self.board.is_game_over()
        if done:
            outcome = self.board.outcome(claim_draw=True)
            if outcome is not None:
                if outcome.termination == chess.Termination.CHECKMATE:
                    # Reward from the perspective of the agent who just moved.
                    if outcome.winner == moving_color:
                        reward += 10  # Win bonus.
                    else:
                        reward -= 10  # Loss penalty.
                else:
                    reward += 0  # Neutral reward for draws/other terminations.

        return self.get_observation(), reward, done, {"illegal_move": illegal}



    def render(self, mode='human'):
        # Create the window and canvas once.
        if not hasattr(self, 'window'):
            self.window = tk.Tk()
            self.window.title("Chess Board")
            self.cell_size = 60  # size of each square in pixels.
            self.canvas = Canvas(self.window, width=8 * self.cell_size, height=8 * self.cell_size)
            self.canvas.pack()

        # Clear previous drawings.
        self.canvas.delete("all")

        # Define board colors.
        light_color = "#F0D9B5"
        dark_color = "#B58863"

        # Mapping from integer board representation to Unicode chess pieces.
        piece_unicode = {
            0: "",
            1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔",
           -1: "♟", -2: "♞", -3: "♝", -4: "♜", -5: "♛", -6: "♚"
        }

        board_array = self.get_observation()

        # Draw each square and piece.
        for row in range(8):
            for col in range(8):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = light_color if (row + col) % 2 == 0 else dark_color
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

                piece_val = board_array[row, col]
                piece = piece_unicode.get(piece_val, "")
                if piece:
                    self.canvas.create_text(x1 + self.cell_size/2,
                                            y1 + self.cell_size/2,
                                            text=piece,
                                            font=("Helvetica", 32))

        self.window.update_idletasks()
        self.window.update()

    def close(self):
        if hasattr(self, 'window'):
            self.window.destroy()
