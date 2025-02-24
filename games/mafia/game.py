"""Mafia game implementation with LLM players."""

from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import random
import re

from base.llm_player import BaseLLMPlayer

class Role(Enum):
    MAFIA = "MAFIA"
    VILLAGER = "VILLAGER"
    DOCTOR = "DOCTOR"
    DETECTIVE = "DETECTIVE"

class Phase(Enum):
    NIGHT = "NIGHT"
    DAY_DISCUSSION = "DAY_DISCUSSION"
    DAY_VOTING = "DAY_VOTING"
    GAME_OVER = "GAME_OVER"

@dataclass
class PlayerState:
    """State for a single player"""
    id: int
    role: Role
    alive: bool = True
    last_protected: Optional[int] = None  # For doctor
    investigation_results: Dict[int, Role] = field(default_factory=dict)  # For detective

@dataclass
class GameState:
    """Tracks the current state of the Mafia game"""
    phase: Phase = Phase.NIGHT
    day: int = 1
    round: int = 1
    players: Dict[int, PlayerState] = field(default_factory=dict)
    mafia_chat: List[str] = field(default_factory=list)
    mafia_votes: Dict[int, int] = field(default_factory=dict)
    day_messages: List[str] = field(default_factory=list)
    votes: Dict[int, int] = field(default_factory=dict)
    night_results: List[str] = field(default_factory=list)
    protected_id: Optional[int] = None
    killed_id: Optional[int] = None

@dataclass
class MafiaConfig:
    """Configuration for Mafia game"""
    run_dir: Path
    max_turns: int = 100
    discussion_rounds: int = 3
    max_retries: int = 3

class MafiaGame:
    """Mafia game manager handling game flow and state"""

    def __init__(self, run_dir: Path, max_turns: int = 100, seed: Optional[int] = None):
        """Initialize mafia game with configuration"""
        self.config = MafiaConfig(run_dir=run_dir, max_turns=max_turns)
        self.state = GameState()
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        
        if seed is not None:
            random.seed(seed)

    def initialize_players(self, players: List[BaseLLMPlayer]) -> None:
        """Assign roles and initialize player states"""
        # Ensure we have exactly 7 players
        if len(players) != 7:
            raise ValueError("Mafia game requires exactly 7 players")
            
        # Assign roles
        roles = [
            Role.MAFIA, Role.MAFIA,  # 2 Mafia
            Role.DOCTOR, Role.DETECTIVE,  # Special roles
            Role.VILLAGER, Role.VILLAGER, Role.VILLAGER  # 3 Villagers
        ]
        random.shuffle(roles)
        
        # Initialize player states
        for player, role in zip(players, roles):
            self.state.players[player.player_id] = PlayerState(
                id=player.player_id,
                role=role
            )

    @staticmethod
    def get_system_prompt() -> Dict[str, str]:
        """Return the initial system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are playing a game of Mafia. You will be assigned a specific role and objective.

IMPORTANT:
1. Think carefully about each action and explain your reasoning
2. After explaining, state your action using exact format: ACTION: (value)

Available Actions:
SAY: (message)      - Express thoughts/communicate (all roles)
VOTE: (id)          - Vote to eliminate (all roles, day only)
KILL: (id)          - Choose night kill target (Mafia only)
SAVE: (id)          - Choose player to protect (Doctor only)
INVESTIGATE: (id)    - Learn player's role (Detective only)

Example thought process:
"I think player 2 is acting suspiciously because they haven't contributed much.
Their silence could indicate they're hiding their Mafia role.

VOTE: (2)

I hope others will consider this evidence as well."

You will receive your specific role, objectives, and available actions next."""
        }

    def get_role_prompt(self, player_id: int) -> Dict[str, str]:
        """Get role-specific prompt for a player"""
        player = self.state.players[player_id]
        
        # Get base role content
        role_content = {
            Role.MAFIA: {
                "objective": "Eliminate town players until Mafia equals or outnumbers Town",
                "actions": {
                    "night": ["KILL: (id)      - Choose a player to eliminate"],
                    "day": ["SAY: (message)  - Share thoughts/defend/accuse",
                           "VOTE: (id)      - Vote for elimination",
                           "VOTE: (0)       - Abstain from voting"]
                }
            },
            Role.VILLAGER: {
                "objective": "Work with Town to identify and eliminate all Mafia",
                "actions": {
                    "day": ["SAY: (message)  - Share thoughts/suspicions",
                           "VOTE: (id)      - Vote for elimination", 
                           "VOTE: (0)       - Abstain from voting"]
                }
            },
            Role.DOCTOR: {
                "objective": "Protect players and help Town eliminate all Mafia\nCannot protect same player twice in a row",
                "actions": {
                    "night": ["SAVE: (id)      - Choose a player to protect"],
                    "day": ["SAY: (message)  - Share thoughts (without revealing role)",
                           "VOTE: (id)      - Vote for elimination",
                           "VOTE: (0)       - Abstain from voting"]
                }
            },
            Role.DETECTIVE: {
                "objective": "Investigate players and help Town eliminate all Mafia\nRemember investigation results",
                "actions": {
                    "night": ["INVESTIGATE: (id)  - Learn if player is Mafia/Town"],
                    "day": ["SAY: (message)     - Share thoughts (careful with role)",
                           "VOTE: (id)         - Vote for elimination",
                           "VOTE: (0)          - Abstain from voting"]
                }
            }
        }[player.role]
        
        # Add Mafia team info if applicable
        mafia_info = ""
        if player.role == Role.MAFIA:
            mafia_ids = [p.id for p in self.state.players.values() 
                        if p.role == Role.MAFIA and p.id != player_id]
            mafia_info = f"\nTeam Members: Player(s) {', '.join(map(str, mafia_ids))}"
        
        # Format actions
        action_text = []
        if "night" in role_content["actions"]:
            action_text.extend(["NIGHT ACTIONS:", *role_content["actions"]["night"]])
        if "day" in role_content["actions"]:
            action_text.extend(["DAY ACTIONS:", *role_content["actions"]["day"]])
        
        return {
            "role": "user",
            "content": f"""You are Player {player_id}, a {player.role.value}

OBJECTIVE:
{role_content['objective']}{mafia_info}

AVAILABLE ACTIONS:
{chr(10).join(action_text)}

STRATEGY TIPS:
- Think through each decision carefully
- Consider how your actions affect both short and long-term game state
- Remember past events and use them in your reasoning
- Stay in character and be strategic about information sharing

When ready to act:
1. Explain your thinking
2. State your action using exact format: ACTION: (value)
3. You may add additional thoughts after your action"""
        }

    def parse_action(self, message: str) -> Optional[Tuple[str, str]]:
        """Parse an action from a message, looking for ACTION: (value) pattern"""
        valid_commands = {"SAY", "VOTE", "KILL", "SAVE", "INVESTIGATE"}
        
        # Split message into lines and look for command patterns
        for line in message.split('\n'):
            line = line.strip()
            # Check each valid command
            for cmd in valid_commands:
                pattern = fr"{cmd}: \((.*?)\)"
                match = re.search(pattern, line)
                if match:
                    value = match.group(1).strip()
                    return (cmd, value)
        return None

    def get_living_players(self) -> Set[int]:
        """Get set of living player IDs"""
        return {pid for pid, p in self.state.players.items() if p.alive}

    def get_living_mafia(self) -> Set[int]:
        """Get set of living Mafia player IDs"""
        return {pid for pid, p in self.state.players.items() 
                if p.alive and p.role == Role.MAFIA}

    def get_living_town(self) -> Set[int]:
        """Get set of living Town player IDs"""
        return {pid for pid, p in self.state.players.items() 
                if p.alive and p.role != Role.MAFIA}

    def check_win_condition(self) -> Optional[Dict[str, str]]:
        """Check if game is over and return winner if any"""
        mafia_count = len(self.get_living_mafia())
        town_count = len(self.get_living_town())
        
        if mafia_count == 0:
            return {
                "winner": "Town",
                "reason": "All Mafia eliminated"
            }
        elif mafia_count >= town_count:
            return {
                "winner": "Mafia",
                "reason": "Mafia equals or outnumbers Town"
            }
        return None

    def get_state_update(self, player_id: int) -> Dict[str, str]:
        """Generate state update message for a player"""
        player = self.state.players[player_id]
        
        # Base state info
        state_parts = [
            f"Current Game State:",
            f"Phase: {self.state.phase.value}",
            f"Day: {self.state.day}",
            f"Round: {self.state.round}/3\n",
            f"Players:",
            f"Living: {', '.join(map(str, self.get_living_players()))}",
            f"Dead: {', '.join(map(str, set(self.state.players.keys()) - self.get_living_players()))}\n"
        ]
        
        # Phase-specific information
        if self.state.phase == Phase.NIGHT:
            if player.role == Role.MAFIA and player.alive:
                state_parts.extend([
                    "Mafia Chat:",
                    *self.state.mafia_chat,
                    "\nCurrent Votes:",
                    *[f"Player {pid} voted for Player {target}" 
                      for pid, target in self.state.mafia_votes.items()],
                    "\nChoose your action:",
                    "- SAY: (message)",
                    "- KILL: (id)"
                ])
            elif player.role == Role.DOCTOR and player.alive:
                state_parts.extend([
                    f"Previous protection: {player.last_protected}",
                    "\nChoose player to protect:",
                    "SAVE: (id)"
                ])
            elif player.role == Role.DETECTIVE and player.alive:
                state_parts.extend([
                    "Previous investigations:",
                    *[f"Player {pid} is {role.value}" 
                      for pid, role in player.investigation_results.items()],
                    "\nChoose player to investigate:",
                    "INVESTIGATE: (id)"
                ])
                
        elif self.state.phase == Phase.DAY_DISCUSSION:
            state_parts.extend([
                "Recent Events:",
                *self.state.night_results,
                f"\nRound {self.state.round} Messages:",
                *self.state.day_messages,
                "\nYour turn to speak:",
                "SAY: (message)"
            ])
            
        elif self.state.phase == Phase.DAY_VOTING:
            state_parts.extend([
                "Discussion Summary:",
                *self.state.day_messages,
                "\nCurrent Votes:",
                *[f"Player {pid} voted for Player {target}" 
                  for pid, target in self.state.votes.items()],
                "\nCast your vote:",
                "VOTE: (id) or VOTE: (0)"
            ])
            
        return {
            "role": "user",
            "content": "\n".join(state_parts)
        }

    def validate_action(self, player_id: int, action: str) -> Dict[str, any]:
        """Validate a player's action"""
        player = self.state.players[player_id]
        
        if not player.alive:
            return {
                "valid": False,
                "message": "Dead players cannot take actions"
            }
            
        # Parse action
        parsed_action = self.parse_action(action)
        if parsed_action is None:
            return {
                "valid": False,
                "message": "Invalid action format. Must be 'COMMAND: (value)'"
            }
            
        command, value = parsed_action
        
        # Validate based on phase and role
        if self.state.phase == Phase.NIGHT:
            if player.role == Role.MAFIA:
                if command == "SAY":
                    return {"valid": True, "command": command, "value": value}
                elif command == "KILL":
                    try:
                        target = int(value)
                        if target not in self.get_living_players():
                            return {
                                "valid": False,
                                "message": "Target must be a living player"
                            }
                        if target in self.get_living_mafia():
                            return {
                                "valid": False,
                                "message": "Cannot target other Mafia members"
                            }
                        return {"valid": True, "command": command, "value": target}
                    except ValueError:
                        return {
                            "valid": False,
                            "message": "Kill target must be a player ID"
                        }
                        
            elif player.role == Role.DOCTOR:
                if command == "SAVE":
                    try:
                        target = int(value)
                        if target not in self.get_living_players():
                            return {
                                "valid": False,
                                "message": "Target must be a living player"
                            }
                        if target == player.last_protected:
                            return {
                                "valid": False,
                                "message": "Cannot protect the same player twice in a row"
                            }
                        return {"valid": True, "command": command, "value": target}
                    except ValueError:
                        return {
                            "valid": False,
                            "message": "Save target must be a player ID"
                        }
                        
            elif player.role == Role.DETECTIVE:
                if command == "INVESTIGATE":
                    try:
                        target = int(value)
                        if target not in self.get_living_players():
                            return {
                                "valid": False,
                                "message": "Target must be a living player"
                            }
                        if target in player.investigation_results:
                            return {
                                "valid": False,
                                "message": "Already investigated this player"
                            }
                        return {"valid": True, "command": command, "value": target}
                    except ValueError:
                        return {
                            "valid": False,
                            "message": "Investigation target must be a player ID"
                        }
                        
        elif self.state.phase in (Phase.DAY_DISCUSSION, Phase.DAY_VOTING):
            if command == "SAY" and self.state.phase == Phase.DAY_DISCUSSION:
                return {"valid": True, "command": command, "value": value}
                
            elif command == "VOTE" and self.state.phase == Phase.DAY_VOTING:
                try:
                    target = int(value)
                    if target != 0 and target not in self.get_living_players():
                        return {
                            "valid": False,
                            "message": "Target must be a living player or 0 to abstain"
                        }
                    return {"valid": True, "command": command, "value": target}
                except ValueError:
                    return {
                        "valid": False,
                        "message": "Vote target must be a player ID or 0"
                    }
                    
        return {
            "valid": False,
            "message": f"Invalid action {command} for your role and phase"
        }

    def process_action(self, player_id: int, action: Dict[str, any]) -> None:
        """Process a validated action"""
        command = action["command"]
        value = action["value"]
        
        if command == "SAY":
            if self.state.phase == Phase.NIGHT:
                self.state.mafia_chat.append(f"Player {player_id}: {value}")
            else:
                self.state.day_messages.append(f"Player {player_id}: {value}")
                
        elif command == "KILL":
            self.state.mafia_votes[player_id] = value
            
        elif command == "SAVE":
            self.state.protected_id = value
            self.state.players[player_id].last_protected = value
            
        elif command == "INVESTIGATE":
            target_role = self.state.players[value].role
            self.state.players[player_id].investigation_results[value] = target_role
            
        elif command == "VOTE":
            self.state.votes[player_id] = value

    def process_night_phase(self) -> None:
        """Process all night actions and generate results"""
        # Reset night state
        self.state.night_results = []
        killed_id = None
        
        # Process Mafia kill
        if self.state.mafia_votes:
            # Check for unanimous vote
            vote_counts = {}
            for target in self.state.mafia_votes.values():
                vote_counts[target] = vote_counts.get(target, 0) + 1
                
            # Kill succeeds if all living mafia voted for same target
            living_mafia = self.get_living_mafia()
            for target, count in vote_counts.items():
                if count == len(living_mafia):
                    killed_id = target
                    break
                    
        # Apply doctor protection
        if killed_id is not None:
            if killed_id == self.state.protected_id:
                self.state.night_results.append("Someone was attacked but saved by the Doctor!")
            else:
                self.state.players[killed_id].alive = False
                self.state.night_results.append(f"Player {killed_id} was killed in the night!")
        else:
            self.state.night_results.append("The night passes peacefully...")
            
        # Reset night actions
        self.state.mafia_votes.clear()
        self.state.protected_id = None
        self.state.mafia_chat.clear()

    def process_voting_phase(self) -> None:
        """Process day phase voting"""
        if not self.state.votes:
            self.state.night_results.append("No one was voted out.")
            return
            
        # Count votes (excluding abstains)
        vote_counts = {}
        for target in self.state.votes.values():
            if target != 0:  # Don't count abstains
                vote_counts[target] = vote_counts.get(target, 0) + 1
                
        if not vote_counts:
            self.state.night_results.append("Everyone abstained. No one was voted out.")
            return
            
        # Find player with majority
        living_count = len(self.get_living_players())
        needed_votes = (living_count // 2) + 1
        
        for target, count in vote_counts.items():
            if count >= needed_votes:
                self.state.players[target].alive = False
                self.state.night_results.append(f"Player {target} was voted out!")
                break
        else:
            self.state.night_results.append("No majority reached. No one was voted out.")
            
        # Reset votes
        self.state.votes.clear()
        self.state.day_messages.clear()

    def advance_phase(self) -> None:
        """Advance to the next game phase"""
        if self.state.phase == Phase.NIGHT:
            self.process_night_phase()
            self.state.phase = Phase.DAY_DISCUSSION
            self.state.round = 1
            
        elif self.state.phase == Phase.DAY_DISCUSSION:
            if self.state.round >= self.config.discussion_rounds:
                self.state.phase = Phase.DAY_VOTING
                self.state.round = 1
            else:
                self.state.round += 1
                
        elif self.state.phase == Phase.DAY_VOTING:
            self.process_voting_phase()
            self.state.phase = Phase.NIGHT
            self.state.day += 1
            self.state.round = 1

    def run(self, players: List[BaseLLMPlayer]) -> Dict[str, any]:
        """Run the mafia game with provided players"""
        # Initialize game
        print("\n=== GAME INITIALIZATION ===")
        print(f"Starting game with {len(players)} players")
        self.initialize_players(players)
        
        # Send initial system prompts and role assignments
        print("\n=== ROLE ASSIGNMENTS ===")
        for player in players:
            print(f"Player {player.player_id} assigned role: {self.state.players[player.player_id].role.value}")
            player.initialize_with_prompt(self.get_system_prompt())
            player.add_message(self.get_role_prompt(player.player_id))
            
        # Game loop
        turn = 0
        while turn < self.config.max_turns:
            print(f"\n=== TURN {turn + 1} ===")
            print(f"Phase: {self.state.phase.value}")
            print(f"Living Players: {sorted(list(self.get_living_players()))}")
            
            # Night Phase
            if self.state.phase == Phase.NIGHT:
                print("\n--- NIGHT PHASE ---")
                print("Mafia, Doctor, and Detective take actions...")
                
                # Reset night actions
                self.state.mafia_votes.clear()
                self.state.protected_id = None
                
                # Get night actions from players
                for player in players:
                    if not self.state.players[player.player_id].alive:
                        continue
                        
                    state = self.get_state_update(player.player_id)
                    response = player.get_response(state)
                    
                    validation = self.validate_action(player.player_id, response)
                    if validation["valid"]:
                        print(f"Player {player.player_id} ({self.state.players[player.player_id].role.value}) action: {response}")
                    else:
                        print(f"Player {player.player_id} failed to take action: {validation['message']}")
                
                # Process night actions
                print("\nNight Results:")
                self.process_night_phase()
                for result in self.state.night_results:
                    print(result)
                
                self.state.phase = Phase.DAY_DISCUSSION
                self.state.day += 1
                
            # Day Discussion Phase
            elif self.state.phase == Phase.DAY_DISCUSSION:
                print("\n--- DAY DISCUSSION PHASE ---")
                print(f"Day {self.state.day} begins...")
                
                # Reset day state
                self.state.day_messages.clear()
                self.state.round = 1
                
                # Three rounds of discussion
                while self.state.round <= 3:
                    print(f"\nDiscussion Round {self.state.round}:")
                    for player in players:
                        if not self.state.players[player.player_id].alive:
                            continue
                            
                        state = self.get_state_update(player.player_id)
                        response = player.get_response(state)
                        validation = self.validate_action(player.player_id, response)
                        
                        if validation["valid"]:
                            print(f"Player {player.player_id}: {response}")
                    
                    self.state.round += 1
                
                self.state.phase = Phase.DAY_VOTING
                
            # Day Voting Phase
            elif self.state.phase == Phase.DAY_VOTING:
                print("\n--- DAY VOTING PHASE ---")
                print("Players cast their votes...")
                
                # Reset votes
                self.state.votes.clear()
                
                # Get votes from players
                for player in players:
                    if not self.state.players[player.player_id].alive:
                        continue
                        
                    state = self.get_state_update(player.player_id)
                    response = player.get_response(state)
                    validation = self.validate_action(player.player_id, response)
                    
                    if validation["valid"]:
                        print(f"Player {player.player_id} voted for: {response}")
                
                # Process votes
                print("\nVoting Results:")
                self.process_voting_phase()
                for result in self.state.night_results:
                    print(result)
                
                self.state.phase = Phase.NIGHT
            
            # Check win condition
            mafia_count = len([p for p in self.state.players.values() 
                             if p.role == Role.MAFIA and p.alive])
            town_count = len([p for p in self.state.players.values() 
                            if p.role != Role.MAFIA and p.alive])
            
            print(f"\nSurvivors - Mafia: {mafia_count}, Town: {town_count}")
            
            result = self.check_win_condition()
            if result:
                print("\n=== GAME OVER ===")
                print(f"{result['winner']} wins! {result['reason']}")
                return {
                    "status": "complete",
                    "winner": result["winner"],
                    "reason": result["reason"],
                    "days": self.state.day,
                    "living_players": self.get_living_players()
                }
                
            turn += 1
            
        print("\n=== GAME OVER ===")
        print(f"Game ended after {self.config.max_turns} turns")
        return {
            "status": "timeout",
            "winner": None,
            "reason": "Maximum turns reached",
            "days": self.state.day,
            "living_players": self.get_living_players()
        }
