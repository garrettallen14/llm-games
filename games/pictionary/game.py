"""Pictionary game implementation with LLM players."""

import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from datetime import datetime
import re
from dataclasses import dataclass, field
import random

from base.llm_player import BaseLLMPlayer

WORD_LIST = [
    # Simple Shapes & Objects
    "moon", "star", "heart", "house", "flag", "clock", "umbrella", "pizza", "snowman", "box",
    
    # Basic Animals
    "cat", "dog", "fish", "bird", "rabbit", "penguin", "duck", "owl", "mouse", "snail",
    
    # Nature Elements
    "tree", "flower", "sun", "mountain", "cloud", "rainbow", "beach", "wave", "leaf", "apple",
    
    # Everyday Items
    "book", "chair", "table", "cup", "hat", "glasses", "key", "lamp", "door", "bell",
    
    # Transportation
    "car", "boat", "ship", "train", "bus", "bike", "plane", "rocket", "truck", "wagon",
    
    # Simple Buildings & Structures
    "igloo", "tent", "bridge", "tower", "castle", "barn", "fence", "well", "ladder", "gate",
    
    # Easy Weather & Sky
    "rain", "snow", "storm", "wind", "star", "moon", "comet", "cloud", "sun", "rainbow",
    
    # Basic Food & Drinks
    "ice cream", "cookie", "cake", "candy", "apple", "banana", "pizza", "egg", "pie", "cup",
    
    # Simple Tools
    "hammer", "saw", "axe", "brush", "pen", "pencil", "ruler", "scissors", "shovel", "bucket"
]

@dataclass
class PictionaryState:
    """Tracks the current state of the Pictionary game."""
    grid_size: int = 100
    grid: np.ndarray = field(init=False)
    colored_pixels: np.ndarray = field(init=False)  # Tracks which pixels have been manually colored
    current_word: str = field(default="")
    guesses: List[str] = field(default_factory=list)
    time_remaining: int = 60
    successful_guesses: Dict[int, List[str]] = field(default_factory=dict)
    players_drawn: set = field(default_factory=set)
    background_color: Tuple[int, int, int] = field(default=(255, 255, 255))
    
    def __post_init__(self):
        """Initialize grid with white background."""
        self.grid = np.full((self.grid_size, self.grid_size, 3), 255, dtype=np.uint8)
        self.colored_pixels = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
    def set_background(self, r: int, g: int, b: int) -> None:
        """Set background color for uncolored pixels."""
        self.background_color = (r, g, b)
        # Only update pixels that haven't been manually colored
        uncolored_mask = ~self.colored_pixels
        self.grid[uncolored_mask] = [r, g, b]
        
    def color_pixel(self, x: int, y: int, r: int, g: int, b: int) -> None:
        """Color a single pixel and mark it as manually colored."""
        self.grid[y, x] = [r, g, b]
        self.colored_pixels[y, x] = True
        
    def color_box(self, x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int) -> None:
        """Color a box region and mark all pixels as manually colored."""
        self.grid[y1:y2+1, x1:x2+1] = [r, g, b]
        self.colored_pixels[y1:y2+1, x1:x2+1] = True
        
    def color_circle(self, x: int, y: int, radius: int, r: int, g: int, b: int) -> None:
        """Color a circle region and mark affected pixels as manually colored."""
        y_indices, x_indices = np.ogrid[:self.grid_size, :self.grid_size]
        distances = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
        circle_mask = distances <= radius
        self.grid[circle_mask] = [r, g, b]
        self.colored_pixels[circle_mask] = True

    def color_line(self, x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int) -> None:
        """Draw a line using Bresenham's algorithm and mark pixels as colored."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        
        step_x = 1 if x2 > x1 else -1
        step_y = 1 if y2 > y1 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.color_pixel(x, y, r, g, b)
                err -= dy
                if err < 0:
                    y += step_y
                    err += dx
                x += step_x
        else:
            err = dy / 2.0
            while y != y2:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.color_pixel(x, y, r, g, b)
                err -= dx
                if err < 0:
                    x += step_x
                    err += dy
                y += step_y
                
        # Color the final point
        if 0 <= x2 < self.grid_size and 0 <= y2 < self.grid_size:
            self.color_pixel(x2, y2, r, g, b)

    def add_successful_guess(self, round_num: int, word: str) -> None:
        """Track a successful guess."""
        if round_num not in self.successful_guesses:
            self.successful_guesses[round_num] = []
        self.successful_guesses[round_num].append(word)

    def all_players_drawn(self, num_players: int) -> bool:
        """Check if all players have had a turn drawing."""
        return len(self.players_drawn) == num_players

@dataclass
class PictionaryConfig:
    """Configuration settings for Pictionary game."""
    run_dir: Path
    max_turns: int = 10
    grid_size: int = 100
    cell_size: int = 8  # Pixel size for display

class PictionaryGame:
    """Pictionary game manager.
    
    A drawing game where an LLM tries to draw a word while others guess.
    """
    
    # Command validation patterns
    COLOR_PATTERN = re.compile(r"COLOR:\s*\((\d+)\s*,\s*(\d+)\)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    BACKGROUND_PATTERN = re.compile(r"BACKGROUND:\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    BOX_FILL_PATTERN = re.compile(r"BOX_FILL:\s*\((\d+)\s*,\s*(\d+)\)\s*\((\d+)\s*,\s*(\d+)\)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    CIRCLE_PATTERN = re.compile(r"CIRCLE:\s*\((\d+)\s*,\s*(\d+)\)\s*(\d+)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    LINE_PATTERN = re.compile(r"LINE:\s*\((\d+)\s*,\s*(\d+)\)\s*\((\d+)\s*,\s*(\d+)\)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    GUESS_PATTERN = re.compile(r"GUESS:\s*(.+)")

    def __init__(self, run_dir: Path, max_turns: int = 10, grid_size: int = 100) -> None:
        """Initialize Pictionary game with configuration."""
        self.config = PictionaryConfig(
            run_dir=run_dir,
            max_turns=max_turns,
            grid_size=grid_size
        )
        self.state = PictionaryState(grid_size=grid_size)
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        self.current_drawer = 0
        self.score = 0
        
    def select_word(self) -> str:
        """Select a random word from the word list."""
        return random.choice(WORD_LIST)

    @staticmethod
    def get_system_prompt() -> Dict[str, str]:
        """Return the system prompt for LLM players."""
        return {
            "role": "system",
            "content": f"""You are playing Pictionary! You will alternate between drawing and guessing.

When DRAWING, you have access to a 100x100 grid and these commands:

COLOR: (x,y) (r,g,b)
Places a single color at position x,y using RGB values (0-255)
Example: COLOR: (int,int) (int,int,int)

BACKGROUND: (r,g,b) 
Sets the background color using RGB values (0-255)
Example: BACKGROUND: (int,int,int)

BOX_FILL: (x1,y1) (x2,y2) (r,g,b)
Colors a rectangular area from (x1,y1) to (x2,y2)
Example: BOX_FILL: (int,int) (int,int) (int,int,int)

CIRCLE: (x,y) radius (r,g,b)
Colors a filled circle with center at (x,y) and given radius
Example: CIRCLE: (int,int) int (int,int,int)

LINE: (x1,y1) (x2,y2) (r,g,b)
Colors a line from (x1,y1) to (x2,y2)
Example: LINE: (int,int) (int,int) (int,int,int)

When GUESSING, use:
GUESS: your_guess_here

Technical Information:
- Grid coordinates: (0,0) at top-left to (99,99) at bottom-right
- Colors persist between drawing commands
- You'll see the current drawing state before each turn
- Multiple commands per turn are allowed
- Invalid commands are skipped
- You must draw the word so that guessers will guess the EXACT word
"""
        }

    def create_grid_image(self) -> str:
        """Generate base64 encoded PNG of current grid state."""
        width = self.config.grid_size * self.config.cell_size
        height = width
        
        image = Image.fromarray(self.state.grid)
        image = image.resize((width, height), Image.NEAREST)
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image.save("board.png", format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def validate_color_command(self, x: int, y: int, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate a color command."""
        if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
            return False, f"Position ({x},{y}) is out of bounds"
        
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def validate_color_values(self, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate RGB color values."""
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def validate_box_fill(self, x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate a box fill command."""
        for x, y in [(x1, y1), (x2, y2)]:
            if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
                return False, f"Position ({x},{y}) is out of bounds"
        
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def validate_circle(self, x: int, y: int, radius: int, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate a circle command."""
        if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
            return False, f"Center position ({x},{y}) is out of bounds"
        
        if radius <= 0:
            return False, "Radius must be positive"
        
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def validate_line(self, x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate a line command."""
        for x, y in [(x1, y1), (x2, y2)]:
            if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
                return False, f"Position ({x},{y}) is out of bounds"
        
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def draw_circle(self, x: int, y: int, radius: int, r: int, g: int, b: int) -> None:
        """Draw a filled circle on the grid."""
        y_indices, x_indices = np.ogrid[:self.config.grid_size, :self.config.grid_size]
        distances = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
        circle_mask = distances <= radius
        self.state.grid[circle_mask] = [r, g, b]

    def check_guess(self, guess: str) -> bool:
        """Check if a guess matches the current word."""
        return guess.lower().strip() == self.state.current_word.lower()

    def attempt_move(self, response: str, is_drawer: bool) -> Dict[str, Any]:
        """Process and execute a move attempt."""
        executed_actions = []
        has_error = False
        error_message = ""
        
        # Handle guessing
        if not is_drawer:
            guess_match = self.GUESS_PATTERN.search(response)
            if guess_match:
                guess = guess_match.group(1).strip()
                self.state.guesses.append(guess)
                if self.check_guess(guess):
                    self.state.is_finished = True
                    self.score += 1
                    return {
                        "valid": True,
                        "message": f"Correct! The word was '{self.state.current_word}'",
                        "end_turn": True
                    }
                return {
                    "valid": True,
                    "message": f"Incorrect guess: {guess}",
                    "end_turn": True
                }
            return {
                "valid": False,
                "message": "Invalid guess format. Use GUESS: your_guess_here",
                "end_turn": False
            }

        # Handle drawing commands
        bg_match = self.BACKGROUND_PATTERN.search(response)
        if bg_match:
            try:
                r, g, b = map(int, bg_match.groups())
                is_valid, error = self.validate_color_values(r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                else:
                    self.state.set_background(r, g, b)
                    executed_actions.append(f"Background set to ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid background format\n"

        for match in self.BOX_FILL_PATTERN.finditer(response):
            try:
                x1, y1, x2, y2, r, g, b = map(int, match.groups())
                is_valid, error = self.validate_box_fill(x1, y1, x2, y2, r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                    continue
                
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                self.state.color_box(x1, y1, x2, y2, r, g, b)
                executed_actions.append(f"Box from ({x1},{y1}) to ({x2},{y2}) filled with ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid box fill format\n"
                continue

        for match in self.CIRCLE_PATTERN.finditer(response):
            try:
                x, y, radius, r, g, b = map(int, match.groups())
                is_valid, error = self.validate_circle(x, y, radius, r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                    continue
                
                self.state.color_circle(x, y, radius, r, g, b)
                executed_actions.append(f"Circle at ({x},{y}) with radius {radius} colored to ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid circle format\n"
                continue

        for match in self.LINE_PATTERN.finditer(response):
            try:
                x1, y1, x2, y2, r, g, b = map(int, match.groups())
                is_valid, error = self.validate_line(x1, y1, x2, y2, r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                    continue
                
                self.state.color_line(x1, y1, x2, y2, r, g, b)
                executed_actions.append(f"Line from ({x1},{y1}) to ({x2},{y2}) colored to ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid line format\n"
                continue

        for match in self.COLOR_PATTERN.finditer(response):
            try:
                x, y, r, g, b = map(int, match.groups())
                is_valid, error = self.validate_color_command(x, y, r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                    continue
                
                self.state.color_pixel(x, y, r, g, b)
                executed_actions.append(f"Cell at ({x},{y}) colored to ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid color format\n"
                continue
        
        if executed_actions:
            return {
                "valid": True,
                "message": "\n".join(executed_actions),
                "end_turn": True
            }
        
        return {
            "valid": False,
            "message": error_message or "Invalid command format",
            "end_turn": False
        }

    def run(self, players: List[BaseLLMPlayer]) -> Dict[str, Any]:
        """Run the game with the LLM players."""
        if len(players) < 2:
            raise ValueError("Need at least 2 players (1 drawer, 1 guesser)")
            
        round_number = 0
        consecutive_errors = 0
        drawing_turns = 0
        
        print("\n=== Starting Pictionary Game ===")
        print(f"Players: {len(players)}")
        print("Models playing:")
        for i, player in enumerate(players, 1):
            print(f"  {i}. {player.model_name}")
        print("\nEach player will get a turn to draw while others guess.\n")
        
        while round_number < self.config.max_turns and (not self.state.all_players_drawn(len(players)) or self.current_drawer == len(players) - 1):
            # Start new round
            if not self.state.current_word:
                self.state.current_word = self.select_word()
                # Reset grid and colored pixels
                self.state.grid = np.full((self.state.grid_size, self.state.grid_size, 3), 255, dtype=np.uint8)
                self.state.colored_pixels = np.zeros((self.state.grid_size, self.state.grid_size), dtype=bool)
                self.state.background_color = (255, 255, 255)  # Reset to white background
                drawing_turns = 0
                self.current_drawer = round_number % len(players)
                current_player = players[self.current_drawer]
                self.state.players_drawn.add(self.current_drawer)
                print(f"\n=== Round {round_number + 1} ===")
                print(f"Drawer: {current_player.model_name}")
                print(f"Word to draw: {self.state.current_word}")
                print("-" * 40)
            
            # Drawing Phase: Same drawer gets 5 turns
            if drawing_turns < 5:
                state_message = {
                    "role": "user",
                    "content": f"You are the DRAWER. Draw the word: {self.state.current_word}\nDrawing Turn {drawing_turns + 1}/5"
                }
                
                try:
                    response = current_player.get_response(state_message, self.create_grid_image())
                    outcome = self.attempt_move(response, True)
                    
                    print(f"Drawing Turn {drawing_turns + 1}/5: {outcome['message']}")
                    
                    if not outcome["valid"]:
                        consecutive_errors += 1
                        current_player.add_message({
                            "role": "system",
                            "content": outcome["message"]
                        })
                        
                        if consecutive_errors >= 3:
                            print("Game ended due to 3 consecutive invalid moves.")
                            break
                    else:
                        consecutive_errors = 0
                    
                    drawing_turns += 1
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error during turn: {str(e)}")
                    
                    if consecutive_errors >= 3:
                        print("Game ended due to 3 consecutive errors.")
                        break
            
            # Guessing Phase: Each non-drawer gets one guess
            else:
                print("\n--- Guessing Phase ---")
                correct_guessers = []
                # Let each non-drawer player make one guess
                for player_idx, player in enumerate(players):
                    # Skip the drawer's turn to guess
                    if player_idx == self.current_drawer:
                        continue
                        
                    state_message = {
                        "role": "user",
                        "content": f"You are a GUESSER. What word is being drawn?\nRound {round_number + 1}"
                    }
                    
                    try:
                        response = player.get_response(state_message, self.create_grid_image())
                        outcome = self.attempt_move(response, False)
                        
                        print(f"{player.model_name}: {outcome['message']}")
                        
                        if not outcome["valid"]:
                            consecutive_errors += 1
                            player.add_message({
                                "role": "system",
                                "content": outcome["message"]
                            })
                            
                            if consecutive_errors >= 3:
                                print("Game ended due to 3 consecutive invalid moves.")
                                break
                        else:
                            consecutive_errors = 0
                            if "Correct!" in outcome["message"]:
                                correct_guessers.append(player.model_name)
                                self.score += 1
                                self.state.add_successful_guess(round_number, self.state.current_word)
                            
                    except Exception as e:
                        consecutive_errors += 1
                        print(f"Error during turn: {str(e)}")
                        
                        if consecutive_errors >= 3:
                            print("Game ended due to 3 consecutive errors.")
                            break
                
                # After all guesses, summarize round
                print(f"\n=== Round {round_number + 1} Summary ===")
                print(f"Word: '{self.state.current_word}'")
                print(f"Drawer: {players[self.current_drawer].model_name}")
                if correct_guessers:
                    print(f"Correctly guessed by: {', '.join(correct_guessers)}")
                else:
                    print("No correct guesses")
                print(f"Models that have drawn: {sorted(players[idx].model_name for idx in self.state.players_drawn)}")
                print("-" * 40)
                
                self.state.current_word = ""  # Reset for next round
                round_number += 1  # Increment round number only after guessing phase
        
        # Game summary
        print("\n=== Game Over! Final Results ===")
        print(f"Total rounds played: {round_number}")
        print("\nCorrect guesses by round:")
        total_correct = 0
        for round_num, words in self.state.successful_guesses.items():
            guessers = len(words)
            total_correct += guessers
            drawer_model = players[round_num % len(players)].model_name
            print(f"Round {round_num + 1}: {guessers} correct guess(es) for word '{words[0]}' (drawn by {drawer_model})")
        print(f"\nTotal correct guesses: {total_correct}")
        print(f"Average correct guesses per round: {total_correct/round_number:.1f}")
        print(f"Models that have drawn: {sorted(players[idx].model_name for idx in self.state.players_drawn)}")
        print("=" * 40)
        
        return {
            "status": "complete",
            "score": self.score,
            "words_guessed": total_correct,
            "total_rounds": round_number,
            "players_drawn": sorted(players[idx].model_name for idx in self.state.players_drawn),
            "successful_guesses": self.state.successful_guesses
        }
