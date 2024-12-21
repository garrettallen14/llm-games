from enum import Enum
from typing import List, Tuple, Optional
import os

class Player(Enum):
    BLACK = "B"
    WHITE = "W"
    EMPTY = " "

class Piece:
    def __init__(self, player: Player, is_king: bool = False):
        self.player = player
        self.is_king = is_king

    def __str__(self):
        symbol = self.player.value
        return symbol.upper() if self.is_king else symbol.lower()

class CheckersGame:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = Player.WHITE
        self.initialize_board()

    def initialize_board(self):
        # Place black pieces
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = Piece(Player.BLACK)

        # Place white pieces
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = Piece(Player.WHITE)

    def is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def get_valid_moves(self, start_row: int, start_col: int) -> List[Tuple[int, int]]:
        if not self.is_valid_position(start_row, start_col):
            return []

        piece = self.board[start_row][start_col]
        if piece is None or piece.player != self.current_player:
            return []

        valid_moves = []
        directions = []
        
        # Regular pieces can only move in one direction
        if piece.player == Player.WHITE and not piece.is_king:
            directions = [(-1, -1), (-1, 1)]  # White moves up
        elif piece.player == Player.BLACK and not piece.is_king:
            directions = [(1, -1), (1, 1)]    # Black moves down
        else:  # Kings can move in all directions
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Check regular moves
        for d_row, d_col in directions:
            new_row, new_col = start_row + d_row, start_col + d_col
            if self.is_valid_position(new_row, new_col) and self.board[new_row][new_col] is None:
                valid_moves.append((new_row, new_col))

        # Check jumps
        for d_row, d_col in directions:
            jump_row, jump_col = start_row + 2*d_row, start_col + 2*d_col
            middle_row, middle_col = start_row + d_row, start_col + d_col
            
            if (self.is_valid_position(jump_row, jump_col) and 
                self.is_valid_position(middle_row, middle_col) and
                self.board[jump_row][jump_col] is None and
                self.board[middle_row][middle_col] is not None and
                self.board[middle_row][middle_col].player != self.current_player):
                valid_moves.append((jump_row, jump_col))

        return valid_moves

    def make_move(self, start_row: int, start_col: int, end_row: int, end_col: int) -> bool:
        if (end_row, end_col) not in self.get_valid_moves(start_row, start_col):
            return False

        # Move the piece
        piece = self.board[start_row][start_col]
        self.board[start_row][start_col] = None
        self.board[end_row][end_col] = piece

        # Handle jumps (capturing)
        if abs(start_row - end_row) == 2:
            middle_row = (start_row + end_row) // 2
            middle_col = (start_col + end_col) // 2
            self.board[middle_row][middle_col] = None

        # King promotion
        if (piece.player == Player.WHITE and end_row == 0) or \
           (piece.player == Player.BLACK and end_row == 7):
            piece.is_king = True

        # Switch players
        self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
        return True

    def display_board(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print("   0 1 2 3 4 5 6 7")
        print("  -----------------")
        for i in range(8):
            print(f"{i} |", end=" ")
            for j in range(8):
                piece = self.board[i][j]
                if piece is None:
                    print("." if (i + j) % 2 == 1 else " ", end=" ")
                else:
                    print(str(piece), end=" ")
            print("|")
        print("  -----------------")
        print(f"Current player: {self.current_player.name}")

    def play(self):
        while True:
            self.display_board()
            
            try:
                # Get move from current player
                start_pos = input(f"{self.current_player.name}'s turn. Enter start position (row col): ")
                if start_pos.lower() == 'quit':
                    break
                    
                start_row, start_col = map(int, start_pos.split())
                
                # Show valid moves for selected piece
                valid_moves = self.get_valid_moves(start_row, start_col)
                if not valid_moves:
                    print("No valid moves for this piece. Try again.")
                    input("Press Enter to continue...")
                    continue
                    
                print("Valid moves:", valid_moves)
                
                end_pos = input("Enter end position (row col): ")
                if end_pos.lower() == 'quit':
                    break
                    
                end_row, end_col = map(int, end_pos.split())
                
                if not self.make_move(start_row, start_col, end_row, end_col):
                    print("Invalid move! Try again.")
                    input("Press Enter to continue...")
                    
            except (ValueError, IndexError):
                print("Invalid input! Please enter row and column numbers (0-7)")
                input("Press Enter to continue...")

if __name__ == "__main__":
    game = CheckersGame()
    game.play()
