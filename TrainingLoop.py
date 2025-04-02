from PythonChessAgent import DQNNet, DQNAgent
from PythonChessEnv import ChessEnv
import chess
import time
import random

if __name__ == "__main__":
    episodes = 100    # Number of training episodes.
    max_steps = 200    # Maximum moves per episode.
    action_size = 20480

    # Create agents for White and Black.
    agent_white = DQNAgent(state_size=(8, 8), action_size=action_size)
    agent_black = DQNAgent(state_size=(8, 8), action_size=action_size)

    env = ChessEnv()

    # Training Loop
    for e in range(episodes):
        state = env.reset()  # Reset the environment (state is an 8x8 array).
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
            # Save the experience and update the model.
            if current_color == chess.WHITE:
                agent_white.remember(prev_state, action, reward, state, done)
                agent_white.replay()
            else:
                agent_black.remember(prev_state, action, reward, state, done)
                agent_black.replay()
            step_count += 1

        print(f"Episode {e+1}/{episodes} finished after {step_count} moves. Epsilon White: {agent_white.epsilon:.2f}, Black: {agent_black.epsilon:.2f}")

    # Save trained models.
    agent_white.save("models/dqn_agent_white.pt")
    agent_black.save("models/dqn_agent_black.pt")

    # ---------------------------
    # Final Game: Agents Play Against Each Other
    # ---------------------------
    # Reset the environment for a new game.
    state = env.reset()

    # Disable exploration: set epsilon to 0 so that agents always choose greedy actions.
    agent_white.epsilon = 0.0
    agent_black.epsilon = 0.0

    # Play until the game is over.
    while not env.board.is_game_over():
        current_color = env.board.turn
        if current_color == chess.WHITE:
            action = agent_white.act(state)
            print("White's turn")
        else:
            action = agent_black.act(state)
            print("Black's turn")

        # Take a step.
        next_state, reward, done, info = env.step(action)
        # If the agent's action is illegal, choose a random legal move.
        if "illegal_move" in info and info["illegal_move"]:
            print("Illegal move chosen")
            legal_moves = list(env.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                base_move = move.from_square * 64 + move.to_square
                # For simplicity, assume no promotion (promotion flag = 0).
                action = base_move  
                next_state, reward, done, info = env.step(action)
        state = next_state

        env.render()
        time.sleep(0.5)

    env.window.mainloop()
