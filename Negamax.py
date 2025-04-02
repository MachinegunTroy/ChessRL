import chess
import numpy as np
import math

pawn_table = np.array([
    [ 0,   0,   0,   0,   0,   0,   0,   0],
    [ 5,   5,   5,  -5,  -5,   5,   5,   5],
    [ 1,   1,   2,   3,   3,   2,   1,   1],
    [ 0.5, 0.5, 1,   2.5, 2.5, 1,   0.5, 0.5],
    [ 0,   0,   0,   2,   2,   0,   0,   0],
    [ 0.5, -0.5,-1,   0,   0,  -1, -0.5, 0.5],
    [ 0.5, 1,   1,  -2,  -2,   1,   1,   0.5],
    [ 0,   0,   0,   0,   0,   0,   0,   0]
])

knight_table = np.array([
    [-5, -4, -3, -3, -3, -3, -4, -5],
    [-4, -2,  0,  0,  0,  0, -2, -4],
    [-3,  0,  1,  1.5, 1.5,  1,  0, -3],
    [-3,  0.5, 1.5, 2,   2, 1.5, 0.5, -3],
    [-3,  0, 1.5, 2,   2, 1.5,  0, -3],
    [-3,  0.5, 1,  1.5, 1.5, 1,  0.5, -3],
    [-4, -2,  0,  0.5, 0.5,  0, -2, -4],
    [-5, -4, -3, -3, -3, -3, -4, -5]
])

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def evaluate_board(board):
    """
    A refined evaluation function that includes:
    - Material balance.
    - Piece-square table bonuses.
    - Simple king safety.
    - Mobility (number of legal moves).
    
    Positive scores favor White, negative scores favor Black.
    """
    score = 0
    
    # Material and positional bonus from piece-square tables.
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            # Positional bonus: use piece-square table if available.
            if piece.piece_type == chess.PAWN:
                table = pawn_table
            elif piece.piece_type == chess.KNIGHT:
                table = knight_table
            else:
                table = np.zeros((8,8))  # Default table for simplicity.
            
            # Convert square index to row, col (from White's perspective).
            row = 7 - (square // 8)
            col = square % 8
            pos_bonus = table[row, col]
            
            if piece.color == chess.WHITE:
                score += value + pos_bonus
            else:
                # For Black, mirror the table.
                score -= value + table[7-row, col]
    
    # King safety: simple bonus if king is castled.
    # We add a bonus if the king is in one of the castled positions.
    # (This is very rudimentary; a real evaluation would be more complex.)
    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        # If castling rights are gone, it's possible the king has moved/castled.
        # For demonstration, add a bonus if White's king is off the central squares.
        white_king_square = board.king(chess.WHITE)
        if white_king_square is not None:
            row, col = 7 - (white_king_square // 8), white_king_square % 8
            if row < 6:  # King is likely castled if not on the first two ranks.
                score += 50
    if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        black_king_square = board.king(chess.BLACK)
        if black_king_square is not None:
            row, col = 7 - (black_king_square // 8), black_king_square % 8
            if row > 1:  # King is likely castled if not on the last two ranks.
                score -= 50

    # Mobility: difference in number of legal moves.
    white_moves = len([m for m in board.legal_moves if board.turn == chess.WHITE])
    board.push(chess.Move.null())  # Simulate turn switch.
    black_moves = len([m for m in board.legal_moves if board.turn == chess.BLACK])
    board.pop()
    score += (white_moves - black_moves) * 5

    return score

def negamax(board, depth, alpha, beta, color):
    """
    Negamax search with alpha-beta pruning.
    
    Args:
        board: a chess.Board() instance.
        depth: current search depth.
        alpha, beta: bounds for alpha-beta pruning.
        color: +1 if the current player is White, -1 if Black.
    
    Returns:
        The evaluated value of the board from the perspective of the current player.
    """
    if depth == 0 or board.is_game_over():
        return color * evaluate_board(board)
    
    max_val = -math.inf
    # Iterate over all legal moves.
    for move in board.legal_moves:
        board.push(move)
        # Recursively evaluate using negamax.
        value = -negamax(board, depth - 1, -beta, -alpha, -color)
        board.pop()
        max_val = max(max_val, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break  # Beta cutoff.
    return max_val

def best_move(board, depth):
    """
    Determines the best move by iterating over legal moves and running negamax.
    
    Args:
        board: a chess.Board() instance.
        depth: search depth.
        
    Returns:
        The move with the highest evaluated value.
    """
    best_val = -math.inf
    best_mv = None
    alpha = -math.inf
    beta = math.inf
    # Set color according to who's turn it is.
    color = 1 if board.turn == chess.WHITE else -1
    for move in board.legal_moves:
        board.push(move)
        # Negamax the subtree, note the minus sign.
        value = -negamax(board, depth - 1, -beta, -alpha, -color)
        board.pop()
        if value > best_val:
            best_val = value
            best_mv = move
        alpha = max(alpha, value)
    return best_mv

if __name__ == '__main__':
    board = chess.Board()
    # Print the starting board.
    print("Starting board:")
    print(board)
    # Search for the best move with a specified depth.
    depth = 10  # For deeper search, increase the depth (but this increases computation time exponentially).
    move = best_move(board, depth)
    print("Best move found:", move)