# games/hangman/game.py

from enum import Enum
from typing import Dict, Optional, List, Set
from pathlib import Path
from datetime import datetime
import re
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from dataclasses import dataclass, field

from base.llm_player import BaseLLMPlayer

class GamePhase(Enum):
    WORD_SELECTION = "word_selection"
    GUESSING = "guessing"

@dataclass
class GameState:
    """Tracks the current state of the Hangman game"""
    current_word: str = ""
    guessed_letters: Set[str] = field(default_factory=set)
    wrong_guesses: int = 0
    games_played: int = 0
    player_scores: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    current_phase: GamePhase = GamePhase.WORD_SELECTION
    move_history: List[str] = field(default_factory=list)
    word_provider_id: int = 1
    guesser_id: int = 2

@dataclass
class HangmanConfig:
    """Configuration for Hangman game"""
    run_dir: Path
    max_turns: int = 100
    wins_required: int = 5
    max_wrong_guesses: int = 6
    min_word_length: int = 3
    max_word_length: int = 20

class HangmanGame:
    def __init__(self, run_dir: Path, max_turns: int = 100):
        self.config = HangmanConfig(
            run_dir=run_dir,
            max_turns=max_turns
        )
        self.state = GameState()
        
        # Regex patterns for move validation
        self.word_pattern = re.compile(r"WORD:\s*([A-Za-z]+)")
        self.guess_pattern = re.compile(r"GUESS:\s*([A-Za-z])")
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": f"""You are playing Hangman to win. Players alternate roles each round.

When Word Provider:
- Choose a word between {self.config.min_word_length}-{self.config.max_word_length} letters long
- Use format 'WORD: X' where X is your word
- English words, but no proper nouns/numbers/special characters

When Guesser:
- Guess one letter at a time using 'GUESS: X' format
- {self.config.max_wrong_guesses} wrong guesses allowed before losing
- Think strategically about common letters
- Example good guesses to start with: E, A, R, I, O, T

First player to win {self.config.wins_required} rounds is the overall winner.
Players swap roles after each round.
When not your turn, just observe the game progress.
You must win the game!"""
        }
        
    def _generate_word_display(self) -> str:
        """Generate the display version of the word with guessed letters revealed"""
        return " ".join(
            letter if letter.upper() in self.state.guessed_letters else "_"
            for letter in self.state.current_word.upper()
        )
        
    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get the current game state from a player's perspective"""
        base_state = (
            f"Games played: {self.state.games_played}\n"
            f"Current scores - Player 1: {self.state.player_scores[1]}, "
            f"Player 2: {self.state.player_scores[2]}\n"
            f"Current roles - Player {self.state.word_provider_id} providing, "
            f"Player {self.state.guesser_id} guessing\n"
        )
        
        if self.state.current_phase == GamePhase.WORD_SELECTION:
            if player_id == self.state.word_provider_id:
                return base_state + "\nYou are the Word Provider. Please provide a word using 'WORD: X' format."
            else:
                return base_state + "\nWaiting for Word Provider to choose a word..."
        else:  # GUESSING phase
            game_state = (
                f"\nWord to guess: {self._generate_word_display()}\n"
                f"Guessed letters: {', '.join(sorted(self.state.guessed_letters)) if self.state.guessed_letters else 'None'}\n"
                f"Wrong guesses remaining: {self.config.max_wrong_guesses - self.state.wrong_guesses}"
            )
            
            if player_id == self.state.word_provider_id:
                return base_state + f"\nWord chosen: {self.state.current_word.upper()}" + game_state + "\nObserving opponent's guesses..."
            else:
                return base_state + game_state + "\nMake your guess using 'GUESS: X' format."
                
    def _draw_game_state(self) -> Image.Image:
        """Create visualization of current game state"""
        img = Image.new('RGB', (800, 600), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("Arial", 36)
        except:
            font = ImageFont.load_default()
            
        # Draw gallows
        draw.line([(100, 500), (500, 500)], fill='black', width=6)  # Base
        draw.line([(300, 500), (300, 100)], fill='black', width=6)  # Pole
        draw.line([(300, 100), (500, 100)], fill='black', width=6)  # Top
        draw.line([(500, 100), (500, 150)], fill='black', width=6)  # Rope
        
        # Draw hangman based on wrong guesses
        if self.state.wrong_guesses >= 1:
            draw.ellipse([(450, 150), (550, 250)], outline='black', width=6)  # Head
        if self.state.wrong_guesses >= 2:
            draw.line([(500, 250), (500, 400)], fill='black', width=6)  # Body
        if self.state.wrong_guesses >= 3:
            draw.line([(500, 300), (400, 350)], fill='black', width=6)  # Left arm
        if self.state.wrong_guesses >= 4:
            draw.line([(500, 300), (600, 350)], fill='black', width=6)  # Right arm
        if self.state.wrong_guesses >= 5:
            draw.line([(500, 400), (400, 500)], fill='black', width=6)  # Left leg
        if self.state.wrong_guesses >= 6:
            draw.line([(500, 400), (600, 500)], fill='black', width=6)  # Right leg
            
        # Draw game state text
        text_y = 50
        draw.text((50, text_y), f"Word: {self._generate_word_display()}", fill='black', font=font)
        text_y += 50
        draw.text((50, text_y), 
                 f"Guessed: {', '.join(sorted(self.state.guessed_letters)) if self.state.guessed_letters else 'None'}", 
                 fill='black', font=font)
        text_y += 50
        draw.text((50, text_y), 
                 f"Wrong guesses left: {self.config.max_wrong_guesses - self.state.wrong_guesses}", 
                 fill='black', font=font)
                 
        return img
        
    def get_game_image(self) -> Optional[str]:
        """Generate base64 encoded PNG of current game state"""
        try:
            img = self._draw_game_state()
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img.save("board.png")
            # img.save(self.config.run_dir / "board.png")
            return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        except Exception as e:
            print(f"Failed to generate game image: {e}")
            return None
            
    def _validate_word(self, word: str) -> bool:
        """Validate if the provided word meets requirements"""
        return (
            word.isalpha() and
            self.config.min_word_length <= len(word) <= self.config.max_word_length and
            word.lower() == word  # Ensure no proper nouns
        )
        
    def _check_win(self) -> bool:
        """Check if the current word has been guessed"""
        return all(letter.upper() in self.state.guessed_letters 
                  for letter in self.state.current_word)
                  
    def _reset_round(self):
        """Reset the game state for a new round"""
        self.state.current_word = ""
        self.state.guessed_letters.clear()
        self.state.wrong_guesses = 0
        self.state.current_phase = GamePhase.WORD_SELECTION
        
        # Swap roles
        self.state.word_provider_id, self.state.guesser_id = (
            self.state.guesser_id,
            self.state.word_provider_id
        )
        
        self.state.move_history.append(
            f"Roles swapped - Provider: Player {self.state.word_provider_id}, "
            f"Guesser: Player {self.state.guesser_id}"
        )
        
    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        print(f"Player {player_id} move: {response}")
        if self.state.current_phase == GamePhase.WORD_SELECTION:
            return self._handle_word_selection(response, player_id)
        else:
            return self._handle_guess(response, player_id)
            
    def _handle_word_selection(self, response: str, player_id: int) -> Dict:
        """Handle word selection phase"""
        if player_id != self.state.word_provider_id:
            return {
                "valid": True,
                "message": f"Waiting for Player {self.state.word_provider_id} to choose a word...",
                "end_turn": True,
                "end_game": False,
                "skip_inference": True
            }
            
        match = self.word_pattern.search(response)
        if not match:
            return {
                "valid": False,
                "message": "Error: Word must be in format 'WORD: X'",
                "end_turn": False
            }
            
        word = match.group(1).lower()
        if not self._validate_word(word):
            return {
                "valid": False,
                "message": f"Error: Word must be {self.config.min_word_length}-{self.config.max_word_length} letters, "
                          "containing only letters, no proper nouns.",
                "end_turn": False
            }
            
        self.state.current_word = word
        self.state.current_phase = GamePhase.GUESSING
        self.state.move_history.append(f"Word selected ({len(word)} letters)")
        
        return {
            "valid": True,
            "message": self.get_current_state(player_id),
            "end_turn": True,
            "end_game": False,
            "skip_inference": (player_id == self.state.word_provider_id)
        }
        
    def _handle_guess(self, response: str, player_id: int) -> Dict:
        """Handle guessing phase"""
        if player_id != self.state.guesser_id:
            return {
                "valid": True,
                "message": self.get_current_state(player_id),
                "end_turn": True,
                "end_game": False,
                "skip_inference": True
            }
            
        match = self.guess_pattern.search(response)
        if not match:
            return {
                "valid": False,
                "message": "Error: Guess must be in format 'GUESS: X'",
                "end_turn": False
            }
            
        guess = match.group(1).upper()
        if guess in self.state.guessed_letters:
            return {
                "valid": False,
                "message": f"Letter {guess} has already been guessed!",
                "end_turn": False
            }
            
        self.state.guessed_letters.add(guess)
        self.state.move_history.append(f"Guessed: {guess}")
        
        # Check if guess is correct
        if guess.lower() not in self.state.current_word.lower():
            self.state.wrong_guesses += 1
            
        # Check win/loss conditions
        game_over = False
        if self._check_win():
            self.state.player_scores[self.state.guesser_id] += 1
            game_over = self.state.player_scores[self.state.guesser_id] >= self.config.wins_required
            self.state.games_played += 1
            message = f"Congratulations! You won! The word was: {self.state.current_word.upper()}"
            self._reset_round()
            
        elif self.state.wrong_guesses >= self.config.max_wrong_guesses:
            self.state.player_scores[self.state.word_provider_id] += 1
            game_over = self.state.player_scores[self.state.word_provider_id] >= self.config.wins_required
            self.state.games_played += 1
            message = f"Game Over! You ran out of guesses. The word was: {self.state.current_word.upper()}"
            self._reset_round()
            
        else:
            message = self.get_current_state(player_id)
            
        return {
            "valid": True,
            "message": message,
            "end_turn": True,
            "end_game": game_over,
            "skip_inference": False
        }
        
    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the game with the provided players"""
        turn = 0
        while turn < self.config.max_turns:
            # Get current player based on game phase
            current_player_id = (
                self.state.word_provider_id if self.state.current_phase == GamePhase.WORD_SELECTION
                else self.state.guesser_id
            )
            current_player = players[current_player_id - 1]
            
            # Get game state
            state_message = {
                "role": "user",
                "content": self.get_current_state(current_player_id)
            }
            
            # Get player response
            try:
                response = current_player.get_response(state_message, self.get_game_image())
                outcome = self.attempt_move(response, current_player_id)
                
                if not outcome["valid"]:
                    continue
                    
                # Notify other player
                other_player_id = 3 - current_player_id
                other_player = players[other_player_id - 1]
                other_state = {
                    "role": "user",
                    "content": self.get_current_state(other_player_id)
                }
                if not outcome["skip_inference"]:
                    other_player.get_response(other_state, self.get_game_image())
                
                if outcome["end_game"]:
                    return self.get_game_result()
                    
            except Exception as e:
                print(f"Error during turn: {str(e)}")
                continue
                
            turn += 1
            
        return self.get_game_result()
        
    def get_game_result(self) -> Dict[str, str]:
        """Get the current game result"""
        self.end_time = datetime.now().isoformat()
        
        if self.state.player_scores[1] >= self.config.wins_required:
            return {
                "status": "complete",
                "winner": "Player 1",
                "player": 1,
                "score": f"{self.state.player_scores[1]}-{self.state.player_scores[2]}",
                "games_played": self.state.games_played,
                "move_history": self.state.move_history
            }
        elif self.state.player_scores[2] >= self.config.wins_required:
            return {
                "status": "complete",
                "winner": "Player 2",
                "player": 2,
                "score": f"{self.state.player_scores[1]}-{self.state.player_scores[2]}",
                "games_played": self.state.games_played,
                "move_history": self.state.move_history
            }
        else:
            return {
                "status": "in_progress",
                "score": f"{self.state.player_scores[1]}-{self.state.player_scores[2]}",
                "games_played": self.state.games_played,
                "move_history": self.state.move_history
            }
            
    def get_final_position(self) -> str:
        """Get string representation of final game state"""
        return (
            f"Final Score: Player 1 {self.state.player_scores[1]} - "
            f"Player 2 {self.state.player_scores[2]}\n"
            f"Games played: {self.state.games_played}\n"
            f"Move history:\n" + "\n".join(self.state.move_history)
        )