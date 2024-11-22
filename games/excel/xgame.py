from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re
import base64
import io
import sys
from games.excel.spreadsheet_utils import (
    create_random_spreadsheet,
    render_spreadsheet,
    print_spreadsheet,
    sort_numbers_to_columns
)

class ExcelGame:
    """Excel spreadsheet manipulation game"""
    
    def __init__(self, run_dir: Path, max_turns: int = 100):
        """Initialize the game"""
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.current_turn = 0
        
        # Initialize spreadsheet
        self.df = create_random_spreadsheet()
        self.solution_df = sort_numbers_to_columns(self.df)
        
        # Compile regex patterns
        self.move_pattern = re.compile(r'^MOVE:\s*([A-I][1-9])\s*,\s*([A-I][1-9])\s*$')
        self.sum_pattern = re.compile(r'^SUM:\s*([A-I][1-9]):([A-I][1-9])\s*$')
        
        # Track game state
        self.game_over = False
        self.messages: List[Dict[str, str]] = []
        self.verified_sums = set()  # Track which columns have had sums verified
        self.last_move = None  # Track the last move for highlighting
    
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the game's system prompt"""
        return {
            "role": "system",
            "content": """You are playing an Excel spreadsheet manipulation game. Your goal is to sort numbers into their matching columns and produce the sums.

Rules:
1. Each number must be moved to its matching column (1's in column A, 2's in column B, etc.)
2. After sorting, verify each column's sum using the SUM command
3. Empty cells are shown as '-'

You can manipulate the spreadsheet by entering commands in the following format:
MOVE: source_cell, target_cell (e.g., 'MOVE: B1, A1' moves B1's value to A1)
SUM: start_cell:end_cell (e.g., 'SUM: A1:A9' sums column A)

Important:

First, I will show you the current board state. Please analyze it and make moves to sort the numbers into their correct columns, then sum the rows in the column I.

To win, you must:
1. Move all numbers to their correct columns (1's in A, 2's in B, etc.)
2. Verify each rows's sum using the SUM command

Invalid commands will return an error message starting with "Error:".
The grid is 9x9 (A1 to I9). Make your moves one at a time, in the exact format:
"MOVE: source_cell, target_cell" or "SUM: start_cell:end_cell".
Avoid overwriting other numbers.
"""
        }
    
    def validate_move(self, source: str, target: str) -> Tuple[bool, str]:
        """Validate a move command"""
        # Check if cells exist
        if not (self._is_valid_cell(source) and self._is_valid_cell(target)):
            return False, "Invalid cell reference"
        
        # Get source value
        source_val = self._get_cell_value(source)
        if source_val == "" or source_val is None:
            return False, f"No number in source cell {source}"
        
        return True, "Valid move"
    
    def validate_sum(self, start: str, end: str) -> Tuple[bool, str]:
        """Validate a sum command"""
        # Check if cells exist
        if not (self._is_valid_cell(start) and self._is_valid_cell(end)):
            return False, "Invalid cell reference"
        
        # Check if in same column
        if start[0] != end[0]:
            return False, "Range must be in the same column"
        
        # Check row order
        start_row = int(start[1])
        end_row = int(end[1])
        if start_row > end_row:
            return False, "Start row must be less than or equal to end row"
        
        return True, "Valid sum range"
    
    def _is_valid_cell(self, cell: str) -> bool:
        """Check if cell reference is valid"""
        return bool(re.match(r'^[A-I][1-9]$', cell))
    
    def _get_cell_value(self, cell: str) -> Optional[str]:
        """Get value from cell"""
        col = ord(cell[0]) - ord('A')
        row = int(cell[1]) - 1
        return str(self.df.iloc[row, col])
    
    def _set_cell_value(self, cell: str, value: str) -> None:
        """Set value in cell"""
        col = ord(cell[0]) - ord('A')
        row = int(cell[1]) - 1
        self.df.iloc[row, col] = value
    
    def process_command(self, command: str) -> str:
        """Process a single command"""
        command = command.strip()
        
        # Check for move command
        move_match = self.move_pattern.match(command)
        if move_match:
            source, target = move_match.groups()
            valid, message = self.validate_move(source, target)
            if valid:
                value = self._get_cell_value(source)
                self._set_cell_value(source, "")
                self._set_cell_value(target, value)
                self.last_move = (source, target)  # Store last move
                self._save_board_image()
                return f"Moved {value} from {source} to {target}"
            return f"Error: {message}"
        
        # Check for sum command
        sum_match = self.sum_pattern.match(command)
        if sum_match:
            start, end = sum_match.groups()
            valid, message = self.validate_sum(start, end)
            if valid:
                col = start[0]
                start_row = int(start[1]) - 1
                end_row = int(end[1]) - 1
                values = [
                    float(x) for x in self.df.iloc[start_row:end_row+1, ord(col)-ord('A')]
                    if x != "" and x is not None
                ]
                if values:
                    self.verified_sums.add(col)
                total = sum(values)
                return f"Sum of {start}:{end} = {int(total)}"
            return f"Error: {message}"
        
        return "Error: Invalid command format. Use MOVE: source, target or SUM: start:end"

    def _save_board_image(self):
        """Save the current board state as an image"""
        img = render_spreadsheet(self.df, last_move=self.last_move)
        img.save("board.png")

    def get_state(self) -> Dict[str, Any]:
        """Get the current game state"""
        print(f"\n📊 Turn {self.current_turn + 1}/{self.max_turns}")
        print(f"Verified columns: {sorted(list(self.verified_sums))}")
        
        # Get ASCII representation
        board_ascii = print_spreadsheet(self.df)
        print("\nCurrent Board State:")
        print(board_ascii)
        
        return {
            "board_ascii": board_ascii,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "verified_columns": sorted(list(self.verified_sums))
        }

    def get_board_image(self) -> Optional[str]:
        """Get base64 encoded PNG of current board state"""
        try:
            img = render_spreadsheet(self.df)
            img_path = "board.png"
            img.save(img_path)
            
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            print(f"Warning: Failed to generate board image: {e}")
            return None

    def run(self, players: List['BaseLLMPlayer']) -> Dict[str, Any]:
        """Run the game"""
        print("\n🎮 Starting Excel Game")
        player = players[0]  # Single player game
        
        # Initialize with system prompt
        system_prompt = self.get_system_prompt()
        self.messages.append(system_prompt)
        
        while not self.game_over and self.current_turn < self.max_turns:
            # Get current state
            state = self.get_state()
            
            # Format board state message
            board_state = (
                f"Current board state (Turn {state['current_turn'] + 1}/{state['max_turns']}):\n"
                f"Goal: Move each number N to column N (1's in A, 2's in B, etc.)\n\n"
                f"{state['board_ascii']}\n\n"
                f"Verified columns: {state['verified_columns']}\n\n"
                f"Enter one command (MOVE: source, target or SUM: start:end)"
            )
            
            # Create state message
            state_message = {
                "role": "user",
                "content": board_state
            }
            self.messages.append(state_message)
            
            # Get player's move with board image
            response = player.get_response(state_message, self.get_board_image())
            print(f"\n🤖 Player: {response}")
            
            # Add player's response to messages
            assistant_message = {
                "role": "assistant",
                "content": response
            }
            self.messages.append(assistant_message)
            
            # Process commands
            commands = response.split('\n')
            for command in commands:
                if command.strip():
                    result = self.process_command(command.strip())
                    print(f"💬 Game: {result}")
                    
                    # Add command result to messages
                    result_message = {
                        "role": "system" if not result.startswith("Error:") else "user",
                        "content": result
                    }
                    self.messages.append(result_message)
            
            # Check if solved
            if self.is_solved():
                print("\n🎉 Game Won!")
                self.game_over = True
                self.messages.append({
                    "role": "system",
                    "content": "🎉 Congratulations! You've won the game!"
                })
                return {"outcome": "success", "turns": self.current_turn + 1}
            
            self.current_turn += 1
        
        if not self.game_over:
            print("\n⏰ Game Over - Max turns reached!")
            self.messages.append({
                "role": "system",
                "content": "⏰ Game Over - Maximum turns reached!"
            })
        
        return {"outcome": "failure", "turns": self.current_turn}

    def is_solved(self) -> bool:
        """Check if the game is solved"""
        print("\n🎯 Checking win condition...")
        # Check each column
        for col_idx, col in enumerate(self.df.columns):
            if col == 'J':  # Skip sum column
                continue
            # Get non-empty values in column
            values = [x for x in self.df[col] if x != "" and x is not None]
            if values:  # Only check columns that have numbers
                # Check if all values match column number
                expected = col_idx + 1
                if not all(int(x) == expected for x in values):
                    print(f"❌ Column {col} has incorrect numbers")
                    return False
                # Check if sum was verified for this column
                if col not in self.verified_sums:
                    print(f"❌ Column {col} sum not verified")
                    return False
                print(f"✅ Column {col} is correct and verified")
        print("🎉 Game solved!")
        return True
