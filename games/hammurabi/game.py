from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import random
import json
from dataclasses import dataclass, field
import re

from base.llm_player import BaseLLMPlayer

@dataclass
class HammurabiState:
    """Tracks the current state of the Hammurabi game"""
    starved: int = 0
    immigrants: int = 5
    population: int = 100
    harvest: int = 3000
    bushels_per_acre: int = 3
    rats_ate: int = 200
    bushels_in_storage: int = 2800
    acres_owned: int = 1000
    cost_per_acre: int = 19
    plague_deaths: int = 0
    year: int = 1
    move_history: List[Dict] = field(default_factory=list)

@dataclass
class HammurabiConfig:
    """Configuration for Hammurabi game"""
    run_dir: Path
    max_turns: int = 10  # Classic game is 10 years
    min_cost_per_acre: int = 17
    max_cost_per_acre: int = 23
    bushels_per_person: int = 20
    acres_per_person: int = 10
    bushels_to_plant: int = 1
    plague_chance: float = 0.15
    rat_chance: float = 0.40
    seed: Optional[int] = None

class HammurabiGame:
    """Implementation of the classic Hammurabi resource management game"""
    
    def __init__(self, run_dir: Path, max_turns: int = 10, seed: Optional[int] = None):
        """Initialize the Hammurabi game"""
        self.config = HammurabiConfig(
            run_dir=run_dir,
            max_turns=max_turns,
            seed=seed  # Pass seed to config
        )
        self.state = HammurabiState()
        self.players: List[BaseLLMPlayer] = []
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.game_over = False
        
        # Retry and move tracking
        self.max_retries = 5
        self.retry_count = 0
        
        # Set random seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def get_system_prompt(self) -> Dict[str, str]:
        """Return a minimal system prompt for LLM players"""
        return {
            "role": "system",
            "content": """Hammurabi Game: Rule Samaria for 10 years.

Goal: Maximize your score.
- Score = Bushels in storage + acres owned * 20 + population * 1000

Core Rules:
- Start: 100 people, 1000 acres, 2800 bushels
- Each person needs 20 bushels/year
- Max 10 acres per person
- 1 bushel plants 1 acre
- If your population goes to 0, you lose the game

Yearly Randomness:
- Harvest: 1-8 bushels/acre
- 15% plague: Halve population
- 40% rats: Halve stored bushels
- Immigration: 0-10 people if no starvation

You may think and plan before submitting your move.
You must submit your final move in the exact following format:

BUY: (+-integer)
FEED: (integer)
PLANT: (integer)

The integers must be placed within parentheses in the exact format.
"""
        }

    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get the minimal current game state"""
        return f"""Year {self.state.year}/10
Population: {self.state.population} ({self.state.starved} starved, {self.state.immigrants} arrived)
Harvest: {self.state.harvest} bushels ({self.state.bushels_per_acre}/acre)
Storage: {self.state.bushels_in_storage} bushels (rats ate {self.state.rats_ate})
Land: {self.state.acres_owned} acres at {self.state.cost_per_acre} bushels/acre
Plague deaths: {self.state.plague_deaths}

Valid moves:
Buy/Sell: {-self.state.acres_owned} to {self.state.bushels_in_storage // self.state.cost_per_acre}
Feed: 0 to {self.state.bushels_in_storage}
Plant: 0 to {min(self.state.acres_owned, self.state.population * 10, self.state.bushels_in_storage)}"""

    @staticmethod
    def extract_integer(key: str, response: str) -> int:
        """
        Extract the last integer found in parentheses after the given key.
        Example: If key is "BUY" and response contains "BUY: (-5)", returns -5
        
        Args:
            key: String to search for (e.g., "BUY", "FEED", "PLANT")
            response: Full text response to search in
        
        Returns:
            Integer value found after the key, or 0 if no match found
        """
        # Look for pattern: key followed by ":" then an integer in parentheses
        pattern = f"{key}:\s*\((-?\d+)\)"
        matches = re.findall(pattern, response)
        
        # Return last match if found, otherwise 0
        return int(matches[-1]) if matches else None

    def validate_move(self, move: Dict) -> tuple[bool, List[str]]:
        """Validate a move based on game rules"""
        errors = []

        # BUYING LAND:
        if move["buy"] > 0:
            # You must have enough grain to pay for land purchases (land cost × acres)
            if self.state.bushels_in_storage < self.state.cost_per_acre * move["buy"]:
                errors.append(f"You must have enough grain to pay for land purchases (land cost × acres)")

        # SELLING LAND:
        if move["buy"] < 0:
            # Cannot sell more land than you own
            if move["buy"] < -self.state.acres_owned:
                errors.append(f"Cannot sell more land than you own")

        # FEEDING:
        # Feed cannot be a negative number
        if move["feed"] < 0:
            errors.append(f"Feed cannot be a negative number")

        # Cannot feed with more than the grain currently in storage
        if move["feed"] > self.state.bushels_in_storage:
            errors.append(f"Cannot feed with more than the grain currently in storage")
        
        # PLANTING:
        # Plant cannot be a negative number
        if move["plant"] < 0:
            errors.append(f"Plant cannot be a negative number")

        if move["plant"] > 0:

            # Cannot plant more acres than you own
            if move["plant"] > self.state.acres_owned:
                errors.append(f"Cannot plant more acres than you own")

            # Cannot plant more than (population * 10) acres
            if move["plant"] > self.state.acres_owned:
                errors.append(f"Cannot plant more acres than you own")

            # Need 1 bushel of grain per acre planted
            if move["plant"] > self.state.bushels_in_storage:
                errors.append(f"Need 1 bushel of grain per acre planted")

            # Cannot use more grain than you have remaining after feeding
            if move["plant"] > self.state.bushels_in_storage - move["feed"]:
                errors.append(f"Cannot use more grain than you have remaining after feeding")

        return len(errors) == 0, errors

    def apply_move(self, move: Dict):
        """Apply a validated move to update the game state"""
        # print(f"\n[MOVE] Processing move: buy={move['buy']}, feed={move['feed']}, plant={move['plant']}")

        # Store initial values for calculations
        initial_population = self.state.population
        initial_storage = self.state.bushels_in_storage
        initial_acres = self.state.acres_owned

        # BUY/SELL LAND:
        self.state.acres_owned += move["buy"]
        self.state.bushels_in_storage -= move["buy"] * self.state.cost_per_acre

        # FEED PEOPLE:
        number_fed = move["feed"] // self.config.bushels_per_person
        self.state.starved = self.state.population - number_fed
        self.state.population -= self.state.starved

        # CALCULATE RANDOM EVENTS
        
        # Calculate immigration
        potential_immigrants = random.randint(0, 10)
        if self.state.starved == 0:
            self.state.immigrants = potential_immigrants
            self.state.population += self.state.immigrants
        else:
            self.state.immigrants = 0

        # Calculate plague and rats
        self.state.plague_deaths = 0
        self.state.rats_ate = 0

        plague_roll = random.random()
        if plague_roll < self.config.plague_chance:
            self.state.plague_deaths = self.state.population // 2
            self.state.population -= self.state.plague_deaths
        
        rat_roll = random.random()
        if rat_roll < self.config.rat_chance:
            self.state.rats_ate = self.state.bushels_in_storage // 2
            self.state.bushels_in_storage -= self.state.rats_ate

        # CALCULATE HARVEST
        self.state.bushels_per_acre = random.randint(1,8)
        self.state.harvest = move["plant"] * self.state.bushels_per_acre
        self.state.bushels_in_storage += self.state.harvest

        # UPDATE LAND PRICE
        self.state.cost_per_acre = random.randint(
            self.config.min_cost_per_acre,
            self.config.max_cost_per_acre
        )
        
        # Record move in history with all required fields
        self.state.move_history.append({
            "year": self.state.year,
            "move": move.copy(),
            "result": {
                "starved": self.state.starved,
                "immigrants": self.state.immigrants,
                "plague_deaths": self.state.plague_deaths,
                "rats_ate": self.state.rats_ate,
                "harvest": self.state.harvest,
                "bushels_per_acre": self.state.bushels_per_acre,
                "population": self.state.population,
                "bushels": self.state.bushels_in_storage,
                "acres": self.state.acres_owned
            }
        })
        
        # Increment year after recording history
        self.state.year += 1

    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        try:
            # Extract move components
            move = {
                "buy": self.extract_integer("BUY", response),
                "feed": self.extract_integer("FEED", response),
                "plant": self.extract_integer("PLANT", response),
            }
            
            # Debug print for extracted move
            print(f"[DEBUG] Extracted move: {move}")
            
            # Check for missing or invalid values
            if None in move.values():
                self.retry_count += 1
                print(f"[DEBUG] Invalid move format. Retry count: {self.retry_count}")
                return {
                    "valid": False,
                    "message": "Invalid move format! Each command must include a number in parentheses.\nThe numbers must be placed within parentheses in the exact format.",
                    "status": "continue",
                    "reason": "invalid_format"
                }
            
            # Validate move
            is_valid, errors = self.validate_move(move)
            if not is_valid:
                print(f"[DEBUG] Move validation failed. Errors: {errors}")
                return {
                    "valid": False,
                    "message": str(errors),
                    "status": "continue",
                    "reason": "invalid_move"
                }
            
            # Apply move if it's valid
            self.apply_move(move)
            
            # Check game end conditions
            game_over = self.state.year >= self.config.max_turns or self.state.population <= 0
            if game_over:
                self.end_time = datetime.now().isoformat()
                self.game_over = True
                print("[DEBUG] Game over conditions met")
            
            return {
                "valid": True,
                "message": f"Move accepted.\n\n{self.get_current_state(player_id)}",
                "status": "game_over" if game_over else "continue",
                "reason": "game_complete" if game_over else None
            }
            
        except Exception as e:
            error_msg = f"Unexpected error processing move: {str(e)}"
            print(f"[DEBUG] Exception in attempt_move: {error_msg}")
            return {
                "valid": False,
                "message": error_msg,
                "status": "continue",
                "reason": "error"
            }

    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the game with the provided players"""
        if not players:
            return {"status": "error", "message": "No players provided"}
            
        self.players = players
        current_player = players[0]
        
        while self.state.year < self.config.max_turns + 1 and self.state.population > 0 and not self.game_over:
            try:
                print(f"\n Year {self.state.year} of {self.config.max_turns}")
                print("=" * 50)
                
                # Reset retry count for this turn
                self.retry_count = 0
                
                # Get initial game state once
                state_message = {
                    "role": "user",
                    "content": self.get_current_state(1)  # Always player 1
                }
                # Get first response
                response = current_player.get_response(state_message)
                
                while self.retry_count < self.max_retries:
                    outcome = self.attempt_move(response, 1)
                    
                    if not outcome["valid"]:
                        self.retry_count += 1
                        
                        if self.retry_count >= self.max_retries:
                            print("[DEBUG] Max retries exceeded")
                            return {
                                "status": "game_over",
                                "reason": "failed_to_move",
                                "score": 0,
                                "final_year": self.state.year,
                                "history": self.state.move_history,
                                "summary": "Your reign has ended - failed to make a valid move after 5 attempts!",
                                "seed": self.config.seed
                            }
                        
                        # Only send the error message for retry attempts
                        error_message = {
                            "role": "user", 
                            "content": f"{outcome['message']}\n\nPlease try again with a valid move to continue playing the game. Attempt {self.retry_count + 1} of {self.max_retries}"
                        }
                        response = current_player.get_response(error_message)
                        continue
                    
                    # Valid move was made
                    if outcome["status"] == "game_over":
                        print("[DEBUG] Game over after valid move")
                        return self.get_game_result()
                    
                    # Break out of retry loop on valid move
                    break

                # Reset retry count on valid move
                self.retry_count = 0

            except Exception as e:
                print(f"[DEBUG] Error during turn: {str(e)}")
                continue
                
        print("[DEBUG] Exited main game loop")
        return self.get_game_result()

    def get_game_result(self) -> Dict:
        """Get the final game result with comprehensive scoring"""
        if self.state.population <= 0:
            return {
                "status": "game_over",
                "reason": "population_extinct",
                "score": 0,
                "final_year": self.state.year,
                "history": self.state.move_history,
                "summary": "Your reign has ended in disaster - the entire population has perished!",
                "seed": self.config.seed,
                "metrics": {
                    "population": 0,
                    "acres": self.state.acres_owned,
                    "bushels": self.state.bushels_in_storage
                }
            }
            
        # Calculate final score with detailed components
        final_metrics = {
            "bushels_score": self.state.bushels_in_storage,
            "land_score": self.state.acres_owned * 20,
            "population_score": self.state.population * 100,
            "population_growth": self.state.population - 100,  # Starting population
            "acres_gained": self.state.acres_owned - 1000,  # Starting acres
            "total_immigrants": sum(
                move.get('result', {}).get('immigrants', 0) 
                for move in self.state.move_history
            ),
            "total_starved": sum(
                move.get('result', {}).get('starved', 0) 
                for move in self.state.move_history
            )
        }
        
        # Calculate final score
        final_score = (
            final_metrics["bushels_score"] +
            final_metrics["land_score"] +
            final_metrics["population_score"]
        )
        
        # Determine performance rating
        if final_score < 5000:
            rating = "Terrible"
            message = "Your incompetent leadership will be remembered with scorn."
        elif final_score < 10000:
            rating = "Poor"
            message = "Your reign was barely tolerable."
        elif final_score < 20000:
            rating = "Fair"
            message = "You showed promise, but your people expected more."
        elif final_score < 40000:
            rating = "Good"
            message = "Your reign brought modest prosperity to Samaria."
        elif final_score < 80000:
            rating = "Excellent"
            message = "Your wise leadership has made Samaria flourish!"
        else:
            rating = "Legendary"
            message = "You will be remembered as one of historys greatest rulers!"
        
        return {
            "status": "complete",
            "years_ruled": self.state.year,
            "score": final_score,
            "rating": rating,
            "message": message,
            "seed": self.config.seed,
            "metrics": final_metrics,
            "final_state": {
                "population": self.state.population,
                "acres": self.state.acres_owned,
                "bushels": self.state.bushels_in_storage
            },
            "history": self.state.move_history
        }

    def get_final_position(self) -> str:
        """Get string representation of final game state"""
        result = self.get_game_result()
        
        if result["status"] == "game_over":
            return (
                " Game Over - Your Reign Has Ended!\n"
                "═══════════════════════════════════\n"
                f"Ruled for: {result['final_year']} years\n"
                f"Reason: {result['summary']}\n\n"
                " Move History:\n" +
                json.dumps(self.state.move_history, indent=2)
            )
        
        return (
            " Final Results of Your Reign\n"
            "═══════════════════════════════\n"
            f"Years Ruled: {result['years_ruled']}\n"
            f"Final Population: {result['final_state']['population']:,} citizens\n"
            f"Land Owned: {result['final_state']['acres']:,} acres\n"
            f"Treasury: {result['final_state']['bushels']:,} bushels\n"
            f"Final Score: {result['score']:,}\n"
            f"Rating: {result['rating']}\n"
            f"\n{result['message']}\n\n"
            " Move History:\n" +
            json.dumps(self.state.move_history, indent=2)
        )

    def get_players(self) -> List[BaseLLMPlayer]:
        """Get the list of players"""
        return self.players
