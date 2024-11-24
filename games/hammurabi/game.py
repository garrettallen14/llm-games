from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import random
import json
from dataclasses import dataclass, field

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

class HammurabiGame:
    """Implementation of the classic Hammurabi resource management game"""
    
    def __init__(self, run_dir: Path, max_turns: int = 10):
        """Initialize the Hammurabi game"""
        self.config = HammurabiConfig(
            run_dir=run_dir,
            max_turns=max_turns
        )
        self.state = HammurabiState()
        self.players: List[BaseLLMPlayer] = []
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are playing Hammurabi, the classic resource management game. You are the ruler of ancient Samaria for a ten year term.
            Your decisions each turn must be provided in this exact format:
            MOVE: {"buy": X, "feed": Y, "plant": Z}
            
            Where:
            - X is acres of land to buy (negative to sell)
            - Y is bushels of grain to feed people
            - Z is acres to plant
            
            Key rules:
            - Each person needs 20 bushels of grain per year to survive
            - Each person can farm at most 10 acres of land
            - It takes 1 bushel of grain to plant an acre
            - Land prices fluctuate yearly
            - Random events like plague and rats may occur
            
            Before moving, please think carefully about resource allocation and then respond with a valid move in the exact format specified.
            Your goal is to grow the population and wealth of your kingdom while minimizing starvation."""
        }
        
    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get the current game state"""
        state = (
            f"O great Hammurabi!\n"
            f"You are in year {self.state.year} of your ten year rule.\n"
            f"In the previous year {self.state.starved} people starved to death.\n"
            f"In the previous year {self.state.immigrants} people entered the kingdom.\n"
            f"The population is now {self.state.population}.\n"
            f"We harvested {self.state.harvest} bushels at {self.state.bushels_per_acre} bushels per acre.\n"
            f"Rats destroyed {self.state.rats_ate} bushels, leaving {self.state.bushels_in_storage} bushels in storage.\n"
            f"The city owns {self.state.acres_owned} acres of land.\n"
            f"Land is currently worth {self.state.cost_per_acre} bushels per acre.\n"
            f"There were {self.state.plague_deaths} deaths from the plague.\n\n"
        )
        
        # Add valid move ranges
        max_buy = self.state.bushels_in_storage // self.state.cost_per_acre
        max_sell = self.state.acres_owned
        max_feed = self.state.bushels_in_storage
        max_plant = min(
            self.state.acres_owned,
            self.state.population * self.config.acres_per_person,
            self.state.bushels_in_storage
        )
        
        state += (
            f"Valid moves:\n"
            f"Buy/Sell land: {-max_sell} to {max_buy} acres\n"
            f"Feed people: 0 to {max_feed} bushels\n"
            f"Plant crops: 0 to {max_plant} acres\n"
            f"Please think carefully about resource allocation and respond with a valid move in the exact format specified.\n"
        )
        
        return state
        
    def validate_move(self, move: Dict) -> tuple[bool, List[str]]:
        """Validate a move and return (is_valid, error_messages)"""
        errors = []
        
        # Check move format
        required_keys = ["buy", "feed", "plant"]
        if not all(key in move for key in required_keys):
            errors.append(f"Move must contain all required keys: {required_keys}")
            return False, errors
            
        # Calculate costs
        land_cost = move["buy"] * self.state.cost_per_acre
        feed_cost = move["feed"]
        plant_cost = move["plant"]
        total_cost = land_cost + feed_cost + plant_cost
        
        # Check if we have enough bushels
        if total_cost > self.state.bushels_in_storage:
            errors.append(
                f"Not enough bushels in storage. Required: {total_cost}, Available: {self.state.bushels_in_storage}\n"
                f"Breakdown:\n"
                f"- Land cost: {land_cost} ({move['buy']} acres at {self.state.cost_per_acre} bushels/acre)\n"
                f"- Feeding cost: {feed_cost} bushels\n"
                f"- Planting cost: {plant_cost} bushels"
            )
        
        # Check land constraints
        min_acres = -self.state.acres_owned
        max_acres = self.state.bushels_in_storage // self.state.cost_per_acre
        if not (min_acres <= move["buy"] <= max_acres):
            errors.append(f"Invalid land purchase. Must be between {min_acres} and {max_acres} acres")
        
        # Check planting constraints
        max_plantable = min(
            self.state.acres_owned + move["buy"],  # Can't plant more acres than we own
            self.state.population * self.config.acres_per_person,  # Each person can farm 10 acres
            self.state.bushels_in_storage - land_cost - feed_cost  # Must have enough grain left to plant
        )
        if not (0 <= move["plant"] <= max_plantable):
            errors.append(f"Invalid planting amount. Must be between 0 and {max_plantable} acres")
        
        # Check feeding constraints
        if not (0 <= move["feed"] <= self.state.bushels_in_storage - land_cost):
            errors.append(
                f"Invalid feeding amount. Must be between 0 and {self.state.bushels_in_storage - land_cost} bushels"
            )
        
        return len(errors) == 0, errors

    def apply_move(self, move: Dict):
        """Apply a validated move to update the game state"""
        # Store the move for history
        self.state.move_history.append({
            "year": self.state.year,
            "move": move.copy(),
            "result": {}
        })
        
        # Apply the move
        self.state.acres_owned += move["buy"]
        self.state.bushels_in_storage -= (
            move["buy"] * self.state.cost_per_acre +  # Land cost
            move["feed"] +  # Feeding cost
            move["plant"]  # Planting cost
        )
        
        # Calculate starvation
        bushels_needed = self.state.population * self.config.bushels_per_person
        self.state.starved = (
            self.state.population 
            if move["feed"] == 0
            else max(0, (bushels_needed - move["feed"]) // self.config.bushels_per_person)
        )
        self.state.population -= self.state.starved
        
        # Calculate harvest
        acres_planted = min(
            move["plant"],
            self.state.acres_owned,
            self.state.population * self.config.acres_per_person
        )
        self.state.bushels_per_acre = random.randint(1, 8)
        self.state.harvest = acres_planted * self.state.bushels_per_acre
        self.state.bushels_in_storage += self.state.harvest
        
        # Random events
        if random.random() < self.config.plague_chance:  # 15% chance of plague
            self.state.plague_deaths = self.state.population // 2
            self.state.population -= self.state.plague_deaths
        else:
            self.state.plague_deaths = 0
            
        # Rats
        if random.random() < self.config.rat_chance:  # 40% chance of rats
            self.state.rats_ate = self.state.bushels_in_storage // 2
            self.state.bushels_in_storage -= self.state.rats_ate
        else:
            self.state.rats_ate = 0
            
        # Immigration
        if self.state.starved == 0:
            self.state.immigrants = random.randint(0, 10)
            self.state.population += self.state.immigrants
        else:
            self.state.immigrants = 0
            
        # Update land price for next turn
        self.state.cost_per_acre = random.randint(
            self.config.min_cost_per_acre,
            self.config.max_cost_per_acre
        )
        self.state.year += 1
        
        # Update move history with results
        self.state.move_history[-1]["result"] = {
            "starved": self.state.starved,
            "immigrants": self.state.immigrants,
            "population": self.state.population,
            "harvest": self.state.harvest,
            "rats_ate": self.state.rats_ate,
            "plague_deaths": self.state.plague_deaths,
            "bushels": self.state.bushels_in_storage,
            "acres": self.state.acres_owned
        }

    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        try:
            # Extract move from response
            move_start = response.find('{')
            move_end = response.find('}') + 1
            if move_start == -1 or move_end == -1:
                error_msg = (
                    "Invalid move format. Your response must contain 'MOVE: ' followed by a JSON object.\n"
                    "Example: MOVE: {\"buy\": int, \"feed\": int, \"plant\": int}"
                )
                self._send_error_feedback(player_id, error_msg)
                return {
                    "valid": False,
                    "message": error_msg,
                    "end_turn": False
                }
            
            try:
                move_str = response[move_start:move_end]
                move = json.loads(move_str)
            except json.JSONDecodeError:
                error_msg = (
                    "Invalid JSON format. Your move must be a valid JSON object.\n"
                    "Example: {\"buy\": int, \"feed\": int, \"plant\": int}"
                )
                self._send_error_feedback(player_id, error_msg)
                return {
                    "valid": False,
                    "message": error_msg,
                    "end_turn": False
                }
            
            # Validate move
            is_valid, errors = self.validate_move(move)
            if not is_valid:
                error_msg = "Invalid move!\n" + "\n".join(f"- {err}" for err in errors)
                self._send_error_feedback(player_id, error_msg)
                return {
                    "valid": False,
                    "message": error_msg,
                    "end_turn": False
                }
            
            # Only apply move if it's valid
            self.apply_move(move)
            
            # Check game end conditions
            game_over = self.state.year > self.config.max_turns or self.state.population <= 0
            if game_over:
                self.end_time = datetime.now().isoformat()
            
            return {
                "valid": True,
                "message": f"Move accepted.\n\n{self.get_current_state(player_id)}",
                "end_turn": True,
                "end_game": game_over,
                "skip_inference": False  # Allow LLM to learn from mistakes
            }
            
        except Exception as e:
            error_msg = f"Unexpected error processing move: {str(e)}"
            self._send_error_feedback(player_id, error_msg)
            return {
                "valid": False,
                "message": error_msg,
                "end_turn": False
            }
    
    def _send_error_feedback(self, player_id: int, error_msg: str) -> None:
        """Send error feedback to the player as a system message"""
        feedback = (
            f"Your last move was invalid. Please try again.\n\n"
            f"Error details:\n{error_msg}\n\n"
            f"Current state:\n{self.get_current_state(player_id)}\n\n"
            f"Remember:\n"
            f"1. Each person needs {self.config.bushels_per_person} bushels to survive\n"
            f"2. Each person can farm {self.config.acres_per_person} acres\n"
            f"3. It takes {self.config.bushels_to_plant} bushel to plant an acre\n"
            f"4. Current land price: {self.state.cost_per_acre} bushels per acre"
        )
        
        # Add message to player's conversation
        players = self.get_players()
        if players and 0 <= player_id - 1 < len(players):
            players[player_id - 1].add_message({
                "role": "system",
                "content": feedback
            })
            
    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the game with the provided players"""
        if not players:
            return {"status": "error", "message": "No players provided"}
            
        self.players = players
        current_player = players[0]  # Single player game
        print(f"\n Starting Hammurabi's reign... (Year {self.state.year} of {self.config.max_turns})\n")
        
        while self.state.year <= self.config.max_turns and self.state.population > 0:
            try:
                print(f"\n Year {self.state.year} of {self.config.max_turns}")
                print("=" * 50)
                
                # Keep trying until we get a valid move
                valid_move = False
                retry_count = 0
                max_retries = 5
                
                # Get initial game state once
                state_message = {
                    "role": "user",
                    "content": self.get_current_state(1)  # Always player 1
                }
                
                # Get first response
                response = current_player.get_response(state_message)
                
                while not valid_move and retry_count < max_retries:
                    outcome = self.attempt_move(response, 1)
                    
                    if not outcome["valid"]:
                        print(f" Invalid move: {outcome['message']}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            return {
                                "status": "game_over",
                                "reason": "failed_to_move",
                                "score": 0,
                                "final_year": self.state.year,
                                "history": self.state.move_history,
                                "summary": "Your reign has ended - failed to make a valid move after 5 attempts!"
                            }
                        
                        # Only send the error message for retry attempts
                        error_message = {
                            "role": "user", 
                            "content": f"Error: {outcome['message']}\nPlease try again with a valid move."
                        }
                        response = current_player.get_response(error_message)
                        continue
                    
                    valid_move = True
                    
                    # Extract and show the move details
                    move_start = response.find('{')
                    move_end = response.find('}') + 1
                    if move_start != -1 and move_end != -1:
                        try:
                            move = json.loads(response[move_start:move_end])
                            print("\n Resource Allocation:")
                            print(f" Land: {'Bought' if move['buy'] > 0 else 'Sold'} {abs(move['buy'])} acres "
                                  f"({move['buy'] * self.state.cost_per_acre} bushels)")
                            print(f" Food: {move['feed']} bushels "
                                  f"(can feed {move['feed'] // 20} people)")
                            print(f" Planting: {move['plant']} acres "
                                  f"({move['plant']} bushels)")
                            total_cost = (move['buy'] * self.state.cost_per_acre + 
                                        move['feed'] + 
                                        move['plant'])
                            print(f" Total cost: {total_cost} bushels")
                            print(f" Remaining in storage: {self.state.bushels_in_storage - total_cost} bushels\n")
                        except:
                            pass
                    
                    # Print year summary after valid move
                    if self.state.year > 1:  # Don't show summary for first year
                        print(f"\n Year {self.state.year} Summary:")
                        print(f" Population: {self.state.population} "
                              f"({'↑' + str(self.state.immigrants) if self.state.immigrants > 0 else '↓' + str(self.state.starved) if self.state.starved > 0 else '→'})")
                        print(f" Harvest: {self.state.harvest} bushels "
                              f"({self.state.bushels_per_acre} per acre)")
                        print(f" Rats ate: {self.state.rats_ate} bushels")
                        print(f"  Plague deaths: {self.state.plague_deaths}")
                        print(f" Treasury: {self.state.bushels_in_storage} bushels")
                        print(f"  Land owned: {self.state.acres_owned} acres")
                        
                        # Calculate score consistently with final scoring
                        current_score = (
                            self.state.bushels_in_storage +
                            (self.state.acres_owned * 20) +
                            (self.state.population * 100)
                        )
                        print(f" Current score: {current_score:,}")
                    
                    if outcome["end_game"]:
                        return self.get_game_result()
                
            except Exception as e:
                print(f" Error during turn: {str(e)}")
                continue
                
        return self.get_game_result()
        
    def get_game_result(self) -> Dict:
        """Get the final game result"""
        if self.state.population <= 0:
            return {
                "status": "game_over",
                "reason": "population_extinct",
                "score": 0,
                "final_year": self.state.year,
                "history": self.state.move_history,
                "summary": "Your reign has ended in disaster - the entire population has perished!"
            }
            
        # Calculate final score
        final_score = (
            self.state.bushels_in_storage +
            (self.state.acres_owned * 20) +
            (self.state.population * 100)
        )
        
        # Calculate performance rating
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
            message = "You will be remembered as one of history's greatest rulers!"
        
        return {
            "status": "complete",
            "years_ruled": self.state.year - 1,
            "final_population": self.state.population,
            "final_acres": self.state.acres_owned,
            "final_bushels": self.state.bushels_in_storage,
            "score": final_score,
            "rating": rating,
            "message": message,
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
            f"Final Population: {result['final_population']:,} citizens\n"
            f"Land Owned: {result['final_acres']:,} acres\n"
            f"Treasury: {result['final_bushels']:,} bushels\n"
            f"Final Score: {result['score']:,}\n"
            f"Rating: {result['rating']}\n"
            f"\n{result['message']}\n\n"
            " Move History:\n" +
            json.dumps(self.state.move_history, indent=2)
        )

    def get_players(self) -> List[BaseLLMPlayer]:
        """Get the list of players"""
        return self.players
