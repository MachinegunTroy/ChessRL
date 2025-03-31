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
        """
        Decodes the action and applies the move.
        Action is an integer in [0, 20479]:
          - promotion_flag = action // 4096 (0 means no promotion; 1,2,3,4 specify promotions)
          - base_move = action % 4096, where:
              from_square = base_move // 64
              to_square   = base_move % 64
        """
        # Decode promotion flag and the underlying move.
        promotion_flag = action // 4096  # 0 to 4.
        base_move = action % 4096         # 0 to 4095.
        from_square = base_move // 64
        to_square = base_move % 64

        # Get the moving piece.
        piece = self.board.piece_at(from_square)

        # Construct the move. If the move is a pawn move reaching promotion rank,
        # use the promotion flag. Otherwise, ignore promotion_flag.
        move = chess.Move(from_square, to_square)  # default move.
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
                # Pawn is moving to the last rank; it must promote.
                # Map promotion_flag to a promotion piece.
                # 1 -> Knight, 2 -> Bishop, 3 -> Rook, 4 -> Queen.
                promotion_mapping = {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}
                # If promotion_flag is 0 (i.e. agent did not choose a promotion),
                # default to queen promotion.
                promotion_piece = promotion_mapping.get(promotion_flag, chess.QUEEN)
                move = chess.Move(from_square, to_square, promotion=promotion_piece)

        # Check for an illegal move.
        if move not in self.board.legal_moves:
            reward = -1.0  # Penalty for illegal moves.
            done = True
            return self.get_observation(), reward, done, {"illegal_move": True}

        # Initialize reward.
        reward = 0.0

        # Reward for capturing: determine what piece is captured.
        if self.board.is_capture(move):
            if self.board.is_en_passant(move):
                if self.board.turn == chess.WHITE:
                    captured_square = move.to_square - 8
                else:
                    captured_square = move.to_square + 8
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
                    # For demonstration, assume agent plays as White.
                    if outcome.winner == chess.WHITE:
                        reward += 10  # Win bonus.
                    elif outcome.winner == chess.BLACK:
                        reward -= 10  # Loss penalty.
                # Other terminations (stalemate, repetition, etc.) could be
                # rewarded neutrally or with a slight penalty/bonus.
                else:
                    reward += 0

        return self.get_observation(), reward, done, {}

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

if __name__ == "__main__":
    episodes = 1000    # Number of training episodes.
    max_steps = 200    # Maximum moves per episode.
    action_size = 20480

    # Create agents for White and Black.
    agent_white = DQNAgent(state_size=(8, 8), action_size=action_size)
    agent_black = DQNAgent(state_size=(8, 8), action_size=action_size)

    env = ChessEnv()

    for e in range(episodes):
        state = env.reset()  # state is an 8x8 array.
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            current_color = env.board.turn  # True for White, False for Black.
            prev_state = state.copy()
            if current_color == chess.WHITE:
                action = agent_white.act(state)
            else:
                action = agent_black.act(state)

            state, reward, done, info = env.step(action)
            if current_color == chess.WHITE:
                agent_white.remember(prev_state, action, reward, state, done)
                agent_white.replay()
            else:
                agent_black.remember(prev_state, action, reward, state, done)
                agent_black.replay()
            step_count += 1

        print(f"Episode {e+1}/{episodes} finished after {step_count} moves. Epsilon White: {agent_white.epsilon:.2f}, Black: {agent_black.epsilon:.2f}")

    # Save trained models.
    agent_white.save("dqn_agent_white.pt")
    agent_black.save("dqn_agent_black.pt")

    # Optional: play a final game with the trained agents.
    state = env.reset()
    while not env.board.is_game_over():
        current_color = env.board.turn
        if current_color == chess.WHITE:
            action = agent_white.act(state)
        else:
            action = agent_black.act(state)
        state, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.5)

    env.window.mainloop()
