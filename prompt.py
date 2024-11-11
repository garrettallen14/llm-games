# prompt.py
import chess
from typing import List, Optional
from collections import defaultdict

# System prompt stays the same
SYSTEM_PROMPT = """You are playing a game of chess. Your goal is to win through careful, strategic play.
Chess Notation Rules:
1. Use Standard Algebraic Notation (SAN)
2. For pawns: just write the destination square (e.g., "e4", "d5")
3. For pieces: use piece letter (N=Knight, B=Bishop, R=Rook, Q=Queen, K=King) + destination (e.g., "Nc6", "Bf5")
4. For captures: include 'x' (e.g., "Nxe4", "exd5")
5. For castling: use "O-O" for kingside or "O-O-O" for queenside
You must always respond in EXACTLY this format:
\'MOVE: <your move in standard algebraic notation>\'"""

def format_move_request(board: chess.Board, pgn: str, color: str, opponent_move: Optional[str] = None) -> str:
    message = []
    if opponent_move:
        message.append(f"Opponent played: {opponent_move}")

    message.extend([
        f"Current position:",
        f"",
        f"{board}",
        f"",
        f"FEN: {board.fen()}",
        f"Game so far: {pgn}",
        f"You are playing as {color}.",
        "Think first and find the best next move.",
        "After deciding, respond EXACTLY in the required format:",
        "\'MOVE: <your move in standard algebraic notation>\'"
    ])
    return "\n".join(message)

def organize_legal_moves(board: chess.Board) -> str:
    """Organize legal moves by piece type"""
    moves_by_piece = defaultdict(list)
    piece_names = {
        chess.PAWN: "Pawns",
        chess.KNIGHT: "Knights",
        chess.BISHOP: "Bishops",
        chess.ROOK: "Rooks",
        chess.QUEEN: "Queens",
        chess.KING: "King"
    }
    
    # Organize moves by piece type
    for move in board.legal_moves:
        move_san = board.san(move)
        if "O-O" in move_san:
            if "O-O-O" == move_san:
                moves_by_piece[chess.KING].insert(0, move_san)  # Put queenside first
            else:
                moves_by_piece[chess.KING].insert(0, move_san)  # Put kingside first
        else:
            piece = board.piece_at(move.from_square)
            if piece:
                moves_by_piece[piece.piece_type].append(move_san)
    
    # Format output
    output = []
    for piece_type in [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
        moves = moves_by_piece[piece_type]
        if moves:
            output.append(f"{piece_names[piece_type]}:")
            for move in sorted(moves):
                output.append(f"* {move}")
            output.append("")
            
    return "\n".join(output)

def format_error_message(failed_move: str, board: chess.Board) -> str:
    """Format error message for illegal move"""
    return f"""YOUR PREVIOUS MOVE ATTEMPT was ILLEGAL!!! Do not select that move again. 

These are your only legal moves:

{organize_legal_moves(board)}

You MUST select one legal move from the list above. If you fail to select a legal move, you risk forfeiting the game. Think through your options from this list then respond with the best move out of these.
YOU MAY NOT USE THE MOVE: {failed_move}... it is ILLEGAL!"""