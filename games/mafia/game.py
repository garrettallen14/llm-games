from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import random
import re
from collections import defaultdict
import io
import base64
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from base.llm_player import BaseLLMPlayer

@dataclass
class GameResult:
    """Result of a game"""
    status: str
    winner: Optional[str]
    metadata: Dict[str, Any]

class GamePhase(Enum):
    """Game phases"""
    SETUP = "setup"
    NIGHT = "night"
    DAY = "day"

class PlayerRole(Enum):
    """Player roles"""
    VILLAGER = "villager"
    MAFIA = "mafia"
    DETECTIVE = "detective"
    DOCTOR = "doctor"

@dataclass
class Player:
    """Tracks individual player state and information"""
    role: PlayerRole
    is_alive: bool = True
    night_action_used: bool = False
    private_messages: List[str] = field(default_factory=list)
    investigation_feedback: Set[int] = field(default_factory=set)

@dataclass
class GameState:
    """Represents the current state of the game"""
    players: Dict[int, Player] = field(default_factory=dict)
    living_players: Set[int] = field(default_factory=set)
    dead_players: Set[int] = field(default_factory=set)
    current_phase: GamePhase = GamePhase.SETUP
    day_count: int = 0
    game_events: List[str] = field(default_factory=list)
    phase_messages: List[str] = field(default_factory=list)
    current_mafia_votes: Dict[int, int] = field(default_factory=dict)
    vote_counts: Dict[int, int] = field(default_factory=dict)
    protected_player: Optional[int] = None
    investigation_results: Dict[int, PlayerRole] = field(default_factory=dict)
    mafia_kill_target: Optional[int] = None
    abstain_count: int = 0
    player_votes: Dict[int, int] = field(default_factory=dict)
    player_actions: Dict[int, str] = field(default_factory=dict)
    discussion_rounds: Dict[int, Dict[int, str]] = field(default_factory=lambda: defaultdict(dict))
    win_condition: Optional[str] = None

    def reset_night_actions(self):
        """Reset all night phase state"""
        self.current_mafia_votes.clear()
        self.mafia_kill_target = None
        self.protected_player = None
        self.player_actions.clear()
        for player in self.players.values():
            player.night_action_used = False

@dataclass
class MafiaConfig:
    """Configuration for Mafia game"""
    run_dir: Path
    max_turns: int = 100
    roles: Dict[PlayerRole, int] = field(default_factory=lambda: {
        PlayerRole.MAFIA: 2,
        PlayerRole.VILLAGER: 3,
        PlayerRole.DOCTOR: 1,
        PlayerRole.DETECTIVE: 1
    })
    min_players: int = 7
    max_players: int = 7
    discussion_rounds: int = 5
    vote_threshold: float = 0.5  # Percentage of votes needed for execution

class MafiaGame:
    """Main game class for Mafia game"""
    def __init__(self, run_dir: Path, max_turns: int = 100):
        self.state = GameState()
        self.config = MafiaConfig(
            run_dir=run_dir,
            max_turns=max_turns
        )
        self.players = []  # List of player objects
        
        # Regex patterns for action parsing
        self.say_pattern = re.compile(r'^SAY:\s*(.+)$', re.IGNORECASE)
        self.vote_pattern = re.compile(r'^VOTE:\s*(\d+)$', re.IGNORECASE)
        self.kill_pattern = re.compile(r'^KILL:\s*(\d+)$', re.IGNORECASE)
        self.save_pattern = re.compile(r'^SAVE:\s*(\d+)$', re.IGNORECASE)
        self.investigate_pattern = re.compile(r'^INVESTIGATE:\s*(\d+)$', re.IGNORECASE)
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None

    def get_system_prompt(self) -> Dict[str, str]:
        """Get the system prompt for the current player"""
        role_prompts = {
            "mafia": """You are a Mafia member in a game of Mafia.
                Your goal is to eliminate all non-Mafia players.
                
                During the night phase:
                - Coordinate with other Mafia members to choose a target
                - Use 'KILL: player_id' to vote for killing a player
                
                During the day phase:
                - Try to avoid suspicion
                - Blend in with villagers
                - Vote to eliminate other players using 'VOTE: player_id'
                
                Win condition: Eliminate all non-Mafia players.""",
            
            "villager": """You are a Villager in a game of Mafia.
                Your goal is to identify and eliminate all Mafia members.
                
                During the day phase:
                - Discuss with other players to identify suspicious behavior
                - Vote to eliminate suspected Mafia members using 'VOTE: player_id'
                
                Win condition: Help eliminate all Mafia members.""",
            
            "detective": """You are the Detective in a game of Mafia.
                Your goal is to investigate players and help eliminate all Mafia members.
                
                During the night phase:
                - Investigate one player using 'INVESTIGATE: player_id'
                - You will learn their true role
                
                During the day phase:
                - Use your investigation results to guide the town
                - Vote to eliminate suspected Mafia members using 'VOTE: player_id'
                
                Win condition: Help eliminate all Mafia members.""",
            
            "doctor": """You are the Doctor in a game of Mafia.
                Your goal is to protect players and help eliminate all Mafia members.
                
                During the night phase:
                - Protect one player using 'SAVE: player_id'
                - Protected players cannot be killed that night
                Note: Cannot protect the same player two nights in a row.
                
                During the day phase:
                - Vote to eliminate suspected Mafia members using 'VOTE: player_id'
                
                Win condition: Help eliminate all Mafia members."""
        }

        # Return default prompt during setup
        if self.state.current_phase == GamePhase.SETUP:
            return {
                "role": "system",
                "content": "You are playing a game of Mafia. Wait for your role to be assigned."
            }

        # Get the current player's role and return appropriate prompt
        player_id = len(self.players)  # Current player being initialized
        if player_id not in self.state.players:
            return {
                "role": "system",
                "content": "You are playing a game of Mafia. Wait for your role to be assigned."
            }

        return {
            "role": "system",
            "content": role_prompts[self.state.players[player_id].role.value]
        }

    def initialize_roles(self, players: List[BaseLLMPlayer]) -> None:
        """Initialize player roles and game state"""
        self.players = players
        num_players = len(players)
        
        # Validate player count
        if num_players != self.config.min_players:
            raise ValueError(f"Expected {self.config.min_players} players, got {num_players}")

        # Create list of roles
        roles = []
        for role, count in self.config.roles.items():
            roles.extend([role] * count)
        
        # Shuffle roles
        random.shuffle(roles)
        
        # Assign roles to players
        print("\n==================================================")
        print("                    GAME SETUP")
        print("==================================================\n")
        
        print("Role Assignments:")
        print("-" * 30)
        
        mafia_team = []
        for i, role in enumerate(roles, 1):
            player = Player(role=role)
            self.state.players[i] = player
            self.state.living_players.add(i)
            print(f"Player {i:2d}: {role.value:8s}")
            
            if role == PlayerRole.MAFIA:
                mafia_team.append(i)
        
        print(f"\nMafia Team: Players {mafia_team}")
        print("=" * 50 + "\n")

    def get_current_state(self, player_id: int) -> str:
        """Get the current game state for a player"""
        player = self.state.players[player_id]
        
        # Base state info
        base_state = f"Day {self.state.day_count}\n"
        base_state += f"Phase: {self.state.current_phase.value}\n"
        base_state += f"Living players: {sorted(list(self.state.living_players))}\n"
        base_state += f"Dead players: {sorted(list(self.state.dead_players))}\n\n"
        
        # Add player-specific info
        base_state += f"\nYou are Player {player_id}\n"
        base_state += f"Your role: {player.role.value}\n"
        
        # Show private messages
        if player.private_messages:
            base_state += "\nPrivate Messages:\n"
            base_state += "\n".join(player.private_messages[-5:]) + "\n"

        # Add discussion history during day phase
        if self.state.current_phase == GamePhase.DAY:
            if self.state.discussion_rounds:
                base_state += "\nRecent Discussion:\n"
                for round_num, round_messages in self.state.discussion_rounds.items():
                    base_state += f"\nRound {round_num}:\n"
                    for pid, message in round_messages.items():
                        base_state += f"Player {pid}: {message}\n"

        # Add game events
        if self.state.game_events:
            base_state += "\nRecent Events:\n"
            base_state += "-" * 20 + "\n"
            for event in self.state.game_events:
                base_state += event + "\n"
        
        return base_state

    def _get_discussion_state(self, player_id: int, round_num: int) -> Dict[str, str]:
        """Get the discussion state for a player"""
        state = self.get_current_state(player_id)
        
        # Add round information
        state += f"\nDiscussion Round {round_num} of {self.config.discussion_rounds}\n"
        
        # Show previous messages from this round
        if self.state.discussion_rounds[round_num]:
            state += "\nMessages this round:\n"
            for pid, msg in self.state.discussion_rounds[round_num].items():
                if pid != player_id:  # Don't show player their own message
                    state += f"Player {pid}: {msg}\n"
        
        state += "\nIt's your turn to speak. Use 'SAY: your message' to contribute to the discussion."
        return {
            "role": "user",
            "content": state
        }

    def _process_discussion_phase(self) -> None:
        """Process the discussion phase"""
        # Show game events at start of discussion
        if self.state.game_events:
            print("\nRecent Events:")
            print("-" * 20)
            for event in self.state.game_events:
                print(event)
            print("\n")
            self.state.game_events.clear()  # Clear events after showing

        # Run 5 rounds of discussion
        for round_num in range(1, self.config.discussion_rounds + 1):
            print(f"\nDiscussion Round {round_num}:")
            print("-" * 20 + "\n")
            
            # Get response from each living player
            for player_id in sorted(self.state.living_players):
                current_player = self.players[player_id - 1]  # Adjust for 0-based indexing
                
                # Get player's message
                state_message = self._get_discussion_state(player_id, round_num)
                response = current_player.get_response(state_message, self.get_game_image())
                
                # Process the response
                say_match = self.say_pattern.match(response)
                if say_match:
                    message = say_match.group(1).strip()
                    self.state.discussion_rounds[round_num][player_id] = message
                    
                    print(f"\nPlayer {player_id:2d} says:")
                    print("-" * 20)
                    print(message + "\n")

    def _handle_player_action(self, player_id: int, action: str) -> Dict:
        """Handle a player's night action"""
        player = self.state.players[player_id]
        
        if not player.is_alive:
            return {
                "valid": False,
                "message": "Dead players cannot take actions",
                "end_turn": False
            }
            
        if player.night_action_used:
            return {
                "valid": False,
                "message": "You have already used your night action",
                "end_turn": False
            }
            
        # Store the action
        self.state.player_actions[player_id] = action
        
        # Mark action as used for non-mafia roles
        if player.role != PlayerRole.MAFIA:
            player.night_action_used = True
            
        return {
            "valid": True,
            "message": "Action recorded",
            "end_turn": True
        }

    def _handle_mafia_coordination(self) -> None:
        """Handle the mafia team coordination and voting process"""
        print("\nMafia Team Coordination:")
        print("-" * 30 + "\n")
        
        # Reset mafia votes at start of coordination
        self.state.current_mafia_votes.clear()
        self.state.mafia_kill_target = None
        
        # Get living mafia members
        living_mafia = [pid for pid, p in self.state.players.items() 
                       if p.role == PlayerRole.MAFIA and p.is_alive]
        if not living_mafia:
            return
            
        # Phase 1: Chat Phase
        print("Chat Phase:")
        print("-" * 20)
        for mafia_id in living_mafia:
            current_player = self.players[mafia_id - 1]
            
            state_message = {
                "role": "user",
                "content": (
                    self.get_current_state(mafia_id) + 
                    "\nCoordinate with your fellow mafia members. Send a chat message using 'SAY: message'"
                )
            }
            response = current_player.get_response(state_message, self.get_game_image())
            
            chat_match = self.say_pattern.match(response)
            if chat_match:
                message = chat_match.group(1).split('\n')[0].strip()
                if "KILL:" not in message.upper():
                    team_message = f"Player {mafia_id:2d}: {message}"
                    print(team_message)
        
        # Phase 2: Voting Phase
        print("\nVoting Phase:")
        print("-" * 20)
        for mafia_id in living_mafia:
            current_player = self.players[mafia_id - 1]
            
            state_message = {
                "role": "user",
                "content": (
                    self.get_current_state(mafia_id) + 
                    "\nNow submit your kill vote using 'KILL: player_id'\n" +
                    "- You must vote to kill someone\n" +
                    "- Both mafia members must agree on the target\n" +
                    "- You cannot kill another mafia member"
                )
            }
            response = current_player.get_response(state_message, self.get_game_image())
            
            vote_match = self.kill_pattern.match(response)
            if vote_match:
                target = int(vote_match.group(1))
                if target in self.state.living_players and target not in living_mafia:
                    self.state.current_mafia_votes[mafia_id] = target
                    print(f"Player {mafia_id:2d} voted to kill Player {target}")
        
        # Results
        print("\nResults:")
        print("-" * 20)
        if len(self.state.current_mafia_votes) == len(living_mafia):
            votes = list(self.state.current_mafia_votes.values())
            if len(set(votes)) == 1:
                self.state.mafia_kill_target = votes[0]
                print(f"Mafia team agreed on killing Player {self.state.mafia_kill_target}")
            else:
                print("Mafia team could not agree on a target. No kill tonight.")
        else:
            print("Not all mafia members voted. No kill tonight.")
        print()

    def _handle_night_actions(self) -> None:
        """Handle night actions for Doctor and Detective"""
        # Process Doctor action
        doctor_id = next((pid for pid, p in self.state.players.items() 
                         if p.role == PlayerRole.DOCTOR and p.is_alive), None)
        if doctor_id:
            current_player = self.players[doctor_id - 1]
            
            # Create doctor state message
            state = self.get_current_state(doctor_id)
            state += "\nDoctor's Night Action Phase:\n"
            state += "You can protect one player from being killed tonight.\n"
            state += "You MUST respond with exactly 'SAVE: X' where X is the player_id you want to protect.\n"
            state += "Example: 'SAVE: X' to protect Player X\n"
            
            state_message = {
                "role": "user",
                "content": state
            }
            
            # Get doctor's action
            response = current_player.get_response(state_message, self.get_game_image())
            
            print(f"\nDoctor Action:")
            print("-" * 20)
            
            # First try exact SAVE pattern
            match = self.save_pattern.match(response)
            if not match:
                # Try finding SAVE anywhere in the response
                match = self.save_pattern.search(response)
            
            if match:
                try:
                    target_id = int(match.group(1))
                    if target_id in self.state.living_players:
                        self.state.protected_player = target_id
                        self.state.players[doctor_id].night_action_used = True
                        print(f"Player {doctor_id} protected Player {target_id}")
                    else:
                        print(f"Player {doctor_id} attempted to protect Player {target_id}, but they are not a valid target")
                except ValueError:
                    print(f"Player {doctor_id} provided an invalid player ID")
            else:
                print(f"Player {doctor_id} did not make a valid SAVE action. Response was: {response}")
        
        # Process Detective action
        detective_id = next((pid for pid, p in self.state.players.items() 
                           if p.role == PlayerRole.DETECTIVE and p.is_alive), None)
        if detective_id:
            current_player = self.players[detective_id - 1]
            
            # Create detective state message
            state = self.get_current_state(detective_id)
            state += "\nDetective's Night Action Phase:\n"
            state += "You can investigate one player to learn their role.\n"
            if self.state.investigation_results:
                state += "\nYour previous investigations:\n"
                for pid, role in self.state.investigation_results.items():
                    state += f"Player {pid}: {role.value}\n"
            state += "\nUse 'INVESTIGATE: player_id' to investigate a player.\n"
            
            state_message = {
                "role": "user",
                "content": state
            }
            
            # Get detective's action
            response = current_player.get_response(state_message, self.get_game_image())
            
            match = self.investigate_pattern.match(response)
            if match:
                target_id = int(match.group(1))
                if target_id in self.state.living_players and target_id != detective_id and target_id not in self.state.investigation_results:
                    target_role = self.state.players[target_id].role
                    self.state.players[detective_id].night_action_used = True
                    print(f"\nDetective Action:")
                    print("-" * 20)
                    print(f"Player {detective_id} investigated Player {target_id}")
                    
                    # Store investigation result
                    self.state.investigation_results[target_id] = target_role
                    self.state.game_events.append("The Detective has investigated someone tonight.")
        print()

    def _process_night_actions(self) -> None:
        """Process all night actions in correct order"""
        print("\nNight Phase Results:")
        print("-" * 30 + "\n")
        
        # Process mafia kill attempt if target was selected
        if self.state.mafia_kill_target is not None:
            target = self.state.players[self.state.mafia_kill_target]
            if self.state.mafia_kill_target == self.state.protected_player:
                print("Kill Attempt: Failed - Target was protected")
                print(f"The doctor successfully protected Player {self.state.protected_player}")
            else:
                print("Kill Attempt: Successful")
                print(f"Player {self.state.mafia_kill_target} was killed")
                target.is_alive = False
                self.state.living_players.remove(self.state.mafia_kill_target)
                self.state.dead_players.add(self.state.mafia_kill_target)
        else:
            print("No kill attempt was made tonight")
        
        print("\nPlayer Status:")
        print("-" * 20)
        print(f"Living players: {sorted(list(self.state.living_players))}")
        print(f"Dead players: {sorted(list(self.state.dead_players))}")
        
        # Process Detective investigation results
        detective_id = next((pid for pid, p in self.state.players.items() 
                           if p.role == PlayerRole.DETECTIVE and p.is_alive), None)
        if detective_id:
            # Find the latest investigation result
            new_investigations = {pid: role for pid, role in self.state.investigation_results.items()
                                if pid not in self.state.players[detective_id].investigation_feedback}
            
            for target_id, target_role in new_investigations.items():
                # Enhanced Detective feedback based on role discovered
                if target_role == PlayerRole.MAFIA:
                    morning_feedback = f"ALERT! Your investigation revealed: Player {target_id} is a MAFIA member!"
                elif target_role == PlayerRole.DOCTOR:
                    morning_feedback = f"Your investigation revealed: Player {target_id} is the DOCTOR - a Town ally!"
                elif target_role == PlayerRole.VILLAGER:
                    morning_feedback = f"Your investigation revealed: Player {target_id} is a VILLAGER - a Town member."
                else:
                    morning_feedback = f"Your investigation revealed: Player {target_id} is a {target_role.value}"
                
                self.state.players[detective_id].private_messages.append(morning_feedback)
                # Track that we've given feedback for this investigation
                self.state.players[detective_id].investigation_feedback = set(self.state.investigation_results.keys())
        
        print("\nPlayer Status:")
        print("-" * 20)
        print(f"Living players: {sorted(list(self.state.living_players))}")
        if self.state.dead_players:
            print(f"Dead players: {sorted(list(self.state.dead_players))}")
        print()

    def _validate_target(self, target_id: int, actor_id: int) -> bool:
        """Validate if a target selection is valid"""
        # Check if target exists and is alive
        if target_id not in self.state.players or target_id not in self.state.living_players:
            return False
            
        # Prevent self-targeting for most actions
        if target_id == actor_id and self.state.players[actor_id].role != PlayerRole.DOCTOR:
            return False
            
        # Prevent mafia targeting other mafia
        if (self.state.players[actor_id].role == PlayerRole.MAFIA and 
            self.state.players[target_id].role == PlayerRole.MAFIA):
            return False
            
        return True

    def _reset_night_actions(self) -> None:
        """Reset all night action flags"""
        # Reset player-specific night action states
        for player in self.state.players.values():
            player.night_action_used = False
            player.private_messages = []
            
        # Reset game state variables
        self.state.protected_player = None
        self.state.current_mafia_votes.clear()
        self.state.mafia_kill_target = None

    def _check_win_condition(self) -> bool:
        """Check if either faction has won"""
        mafia_count = sum(1 for p in self.state.players.values() 
                        if p.role == PlayerRole.MAFIA and p.is_alive)
                        
        town_count = sum(1 for p in self.state.players.values()
                        if p.role != PlayerRole.MAFIA and p.is_alive)

        if mafia_count == 0:
            self.state.win_condition = "Town wins! All mafia eliminated."
            return True
            
        if mafia_count >= town_count:
            self.state.win_condition = "Mafia wins! They equal or outnumber the town."
            return True
            
        return False

    def _get_vote_results_announcement(self) -> str:
        """Format voting results announcement"""
        results = ["=== Voting Results ==="]
        
        # Show who voted for whom
        vote_details = []
        for voter_id, target_id in self.state.player_votes.items():
            if target_id == 0:
                vote_details.append(f"Player {voter_id} abstained")
            else:
                vote_details.append(f"Player {voter_id} voted for Player {target_id}")
        if vote_details:
            results.extend(vote_details)
        
        # Show final vote tallies
        results.append("\nFinal Vote Count:")
        for player_id, votes in self.state.vote_counts.items():
            results.append(f"Player {player_id}: {votes} votes")
        if self.state.abstain_count > 0:
            results.append(f"Abstained: {self.state.abstain_count} players")
            
        # Add elimination result
        if self.state.game_events:
            results.append(f"\nResult: {self.state.game_events[-1]}")
            
        return "\n".join(results)

    def _process_elimination(self) -> None:
        """Process voting results and eliminate player if clear majority"""
        print("\nVoting Results:")
        print("-" * 30)
        
        total_votes = sum(self.state.vote_counts.values())
        votes_needed = len(self.state.living_players) // 2 + 1
        
        print(f"Vote counts: {dict(self.state.vote_counts)}")
        print(f"Abstained: {len(self.state.living_players) - total_votes}")
        print(f"Votes needed for majority: {votes_needed}")
        
        if total_votes < len(self.state.living_players):
            print(f"Warning: Only {total_votes} out of {len(self.state.living_players)} living players voted")
            print()
            return
            
        if not self.state.vote_counts:
            print("No votes cast.")
            print()
            return
            
        # Find player with most votes
        max_votes = max(self.state.vote_counts.values())
        players_with_max = [pid for pid, votes in self.state.vote_counts.items() 
                          if votes == max_votes]
                          
        if len(players_with_max) > 1:
            print(f"Tie between players {players_with_max}. No one was eliminated.")
            print()
            return
            
        eliminated_id = players_with_max[0]
        if self.state.vote_counts[eliminated_id] >= votes_needed:
            self.state.players[eliminated_id].is_alive = False
            self.state.living_players.remove(eliminated_id)
            self.state.dead_players.add(eliminated_id)
            print(f"\nPlayer {eliminated_id} was eliminated by town vote!")
            print(f"They were a {self.state.players[eliminated_id].role.value}")
            self.state.game_events.append(f"Player {eliminated_id} was eliminated by town vote!")
        else:
            print("No majority reached. No one was eliminated.")
            print()
            
    def _start_discussion(self) -> None:
        """Start the day discussion phase"""
        self.state.current_phase = GamePhase.DAY
        self.state.day_count += 1
        self._reset_votes()
        
        # Clear discussion tracking for new round
        self.state.discussion_rounds.clear()

    def _handle_discussion(self, response: str, player_id: int, round_num: int) -> Dict:
        """Process discussion phase actions"""
        if not self.state.players[player_id].is_alive:
            return {
                "valid": False,
                "message": "Dead players cannot participate in discussion.",
                "end_turn": True
            }

        match = self.say_pattern.match(response)
        if not match:
            return {
                "valid": False,
                "message": "Invalid discussion format. Use 'SAY: your_message'",
                "end_turn": False
            }

        message = match.group(1).strip()
        if not message:
            return {
                "valid": False,
                "message": "Message cannot be empty",
                "end_turn": False
            }

        # Store message for this specific round
        self.state.discussion_rounds[round_num][player_id] = message

        return {
            "valid": True,
            "message": "Message recorded.",
            "end_turn": True
        }

    def _process_discussion_phase(self) -> None:
        """Process the discussion phase"""
        print("\n==================================================")
        print("                    DAY DISCUSSION")
        print("==================================================\n")

        # Show recent events at start of day
        if len(self.state.game_events) > 0:
            print("Recent Events:")
            print("-" * 20)
            for event in self.state.game_events[-3:]:  # Show last 3 events
                print(event)
            print("\n")
            self.state.game_events.clear()  # Clear events after showing

        # Run 5 rounds of discussion
        for round_num in range(1, self.config.discussion_rounds + 1):
            print(f"\nDiscussion Round {round_num}:")
            print("-" * 20 + "\n")
            
            # Get response from each living player
            for player_id in sorted(self.state.living_players):
                current_player = self.players[player_id - 1]  # Adjust for 0-based indexing
                
                # Get player's message
                state_message = self._get_discussion_state(player_id, round_num)
                response = current_player.get_response(state_message, self.get_game_image())
                
                # Process the response
                say_match = self.say_pattern.match(response)
                if say_match:
                    message = say_match.group(1).strip()
                    self.state.discussion_rounds[round_num][player_id] = message
                    
                    print(f"\nPlayer {player_id:2d} says:")
                    print("-" * 20)
                    print(message + "\n")

    def _get_discussion_state(self, player_id: int, round_num: int) -> Dict[str, str]:
        """Get the discussion state for a player"""
        state = self.get_current_state(player_id)
        
        # Add round information
        state += f"\nDiscussion Round {round_num} of {self.config.discussion_rounds}\n"
        
        # Show previous messages from this round
        if self.state.discussion_rounds[round_num]:
            state += "\nMessages this round:\n"
            for pid, msg in self.state.discussion_rounds[round_num].items():
                if pid != player_id:  # Don't show player their own message
                    state += f"Player {pid}: {msg}\n"
        
        state += "\nIt's your turn to speak. Use 'SAY: your message' to contribute to the discussion."
        return {
            "role": "user",
            "content": state
        }

    def _handle_vote(self, response: str, player_id: int) -> Dict:
        """Process voting phase actions with retry logic"""
        if not self.state.players[player_id].is_alive:
            return {
                "valid": False,
                "message": "Dead players cannot vote.",
                "end_turn": True
            }
            
        match = self.vote_pattern.match(response)
        if not match:
            return {
                "valid": False,
                "message": "Invalid vote format. Use 'VOTE: player_id' or 'VOTE: 0' to abstain.",
                "end_turn": False
            }
            
        target_id = int(match.group(1))
        
        # Validate vote target
        if target_id != 0 and (target_id not in self.state.living_players or target_id == player_id):
            return {
                "valid": False,
                "message": "Invalid target. Choose a living player other than yourself, or 0 to abstain.",
                "end_turn": False
            }
        
        # Record the vote
        if target_id == 0:
            print(f"\nPlayer {player_id:2d} abstained from voting.")
            self.state.abstain_count += 1
            self.state.player_votes[player_id] = 0
        else:
            print(f"\nPlayer {player_id:2d} voted for Player {target_id}.")
            self.state.vote_counts[target_id] = self.state.vote_counts.get(target_id, 0) + 1
            self.state.player_votes[player_id] = target_id
        
        return {
            "valid": True,
            "message": "Vote recorded.",
            "end_turn": True
        }

    def _reset_votes(self) -> None:
        """Reset all voting trackers"""
        self.state.vote_counts = {}  # Reset vote counts
        self.state.player_votes = {}  # Reset player vote tracking
        
        # Reset individual player vote tracking
        for player in self.state.players.values():
            player.private_messages = []

    def _process_voting_phase(self) -> None:
        """Process the voting phase"""
        print("\n==================================================")
        print("                    VOTING PHASE")
        print("==================================================\n")

        # Reset voting state
        self._reset_votes()

        # Each living player must vote
        for player_id in sorted(self.state.living_players):
            current_player = self.players[player_id - 1]
            
            # Create voting state message
            state = self.get_current_state(player_id)
            state += "\nVoting Phase:\n"
            state += "Use 'VOTE: player_id' to vote for elimination (VOTE: 0 to abstain)\n"
            
            if self.state.vote_counts:
                state += "\nCurrent votes:\n"
                for target, count in self.state.vote_counts.items():
                    if target == 0:
                        state += f"Abstain: {count} votes\n"
                    else:
                        state += f"Player {target}: {count} votes\n"
            
            # Get and process vote
            message = {
                "role": "user",
                "content": state
            }
            response = current_player.get_response(message, self.get_game_image())
            self._handle_vote(response, player_id)

        # Process results
        print("\nVoting Results:")
        print("-" * 20)
        
        # Calculate results
        total_votes = len(self.state.player_votes)
        if total_votes == 0:
            print("No votes were cast!")
            return

        # Calculate majority threshold
        living_count = len(self.state.living_players)
        majority_needed = (living_count // 2) + 1
        print(f"Votes needed for majority: {majority_needed}")

        # Find the player with the most votes
        if self.state.vote_counts:
            max_votes = max(self.state.vote_counts.values())
            targets = [pid for pid, votes in self.state.vote_counts.items() 
                      if votes == max_votes and pid != 0]  # Exclude abstain

            if len(targets) == 1 and max_votes >= majority_needed:
                eliminated_id = targets[0]
                self.state.players[eliminated_id].is_alive = False
                self.state.living_players.remove(eliminated_id)
                self.state.dead_players.add(eliminated_id)
                
                print(f"\nPlayer {eliminated_id} was eliminated by town vote!")
                print(f"They were a {self.state.players[eliminated_id].role.value}")
                self.state.game_events.append(f"Player {eliminated_id} was eliminated by town vote!")
            else:
                print("\nNo majority reached. No one was eliminated.")

        print("\nVoting Summary:")
        for player_id, vote in self.state.player_votes.items():
            if vote == 0:
                print(f"Player {player_id} abstained")
            else:
                print(f"Player {player_id} voted for Player {vote}")

    def get_game_result(self) -> GameResult:
        """Get the final game result"""
        self.end_time = datetime.now().isoformat()
        
        living_mafia = [pid for pid, p in self.state.players.items()
                       if p.role == PlayerRole.MAFIA and p.is_alive]
        living_town = [pid for pid, p in self.state.players.items()
                       if p.role != PlayerRole.MAFIA and p.is_alive]
        
        return GameResult(
            status="complete",
            winner=self.state.win_condition,
            metadata={
                "days_elapsed": self.state.day_count,
                "living_mafia": living_mafia,
                "living_town": living_town,
                "game_events": self.state.game_events,
                "final_state": self.get_current_state(1)
            }
        )

    def _create_game_image(self) -> Image.Image:
        """Create visualization of current game state"""
        width = 800
        height = 600
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("Arial", 20)
        except:
            font = ImageFont.load_default()

        # Draw player circle
        center_x = width // 2
        center_y = height // 2
        radius = 200
        player_count = len(self.state.players)
        
        for player_id in range(1, player_count + 1):
            angle = 2 * 3.14159 * player_id / player_count
            x = center_x + int(radius * np.cos(angle))
            y = center_y + int(radius * np.sin(angle))
            
            # Draw player node
            color = 'gray' if player_id in self.state.dead_players else 'green'
            draw.ellipse([x-20, y-20, x+20, y+20], fill=color)
            draw.text((x, y), str(player_id), fill='black', font=font, anchor="mm")

        # Draw phase information
        draw.text((10, 10), f"Phase: {self.state.current_phase.value}", fill='black', font=font)
        draw.text((10, 40), f"Day: {self.state.day_count}", fill='black', font=font)
        
        return img

    def get_game_image(self) -> Optional[str]:
        """Generate base64 encoded PNG of current game state"""
        try:
            img = self._create_game_image()
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img.save("board.png")
            return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        except Exception as e:
            print(f"Failed to generate game image: {e}")
            return None

    def _process_night_phase(self) -> None:
        """Process the entire night phase in order"""
        print("\n==================================================")
        print("                    NIGHT PHASE")
        print("==================================================\n")

        # Reset night action states
        self._reset_night_actions()

        # Step 1: Mafia team coordination and kill vote
        self._process_mafia_night_phase()

        # Step 2: Doctor and Detective actions
        print("\nSpecial Role Actions:")
        print("-" * 30)
        self._handle_night_actions()

        # Step 3: Process all actions in correct order
        self._process_night_actions()

        # Check win conditions after night phase
        self._check_win_condition()

    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Main game loop with proper ordering"""
        self.players = players
        
        self.initialize_roles(players)
        
        turn = 0
        
        while True:
            # Night Phase
            self._process_night_phase()
            
            if self._check_win_condition():
                break
                
            # Day Phase
            self.state.current_phase = GamePhase.DAY
            self.state.day_count += 1
            
            # Discussion Phase
            self._process_discussion_phase()  # No round parameter needed anymore
            
            # Voting Phase
            self._process_voting_phase()
            
        return self.get_game_result()

    def _process_mafia_night_phase(self) -> None:
        """Process the mafia team's night phase actions"""
        print("\nMafia Team Coordination:")
        print("-" * 30 + "\n")
        
        # Reset mafia votes at start of coordination
        self.state.current_mafia_votes.clear()
        self.state.mafia_kill_target = None
        
        # Get living mafia members
        living_mafia = [pid for pid, player in self.state.players.items() 
                       if player.role == PlayerRole.MAFIA and player.is_alive]
        
        if not living_mafia:
            return
            
        # Phase 1: Chat Phase
        print("Chat Phase:")
        print("-" * 20)
        
        # Store messages for this round
        mafia_messages = []
        
        # Each mafia member gets to send up to 3 messages
        for _ in range(3):
            for mafia_id in living_mafia:
                current_player = self.players[mafia_id - 1]
                
                # Create mafia-specific state message
                state = self.get_current_state(mafia_id)
                state += "\nMafia Team Chat Phase:\n"
                state += f"Your mafia teammate(s): Players {living_mafia}\n"
                if mafia_messages:
                    state += "\nPrevious messages:\n"
                    state += "\n".join(mafia_messages)
                state += "\n\nUse 'SAY: message' to chat with your team."
                
                state_message = {
                    "role": "user",
                    "content": state
                }
                
                response = current_player.get_response(state_message, self.get_game_image())
                
                # First check for SAY command
                chat_match = self.say_pattern.match(response)
                if chat_match:
                    message = chat_match.group(1).strip()
                    if "KILL:" not in message.upper():
                        team_message = f"Player {mafia_id}: {message}"
                        mafia_messages.append(team_message)
                        print(team_message)
                
                # Then check for KILL command anywhere in the response
                kill_match = self.kill_pattern.search(response)  # Use search instead of match
                if kill_match:
                    target = int(kill_match.group(1))
                    if target in self.state.living_players and target not in living_mafia:
                        self.state.current_mafia_votes[mafia_id] = target
                        print(f"Player {mafia_id} voted to kill Player {target}")
        
        print("\nVoting Phase:")
        print("-" * 20)
        
        # Phase 2: Voting Phase - Each mafia member must vote
        for mafia_id in living_mafia:
            current_player = self.players[mafia_id - 1]
            
            # Create mafia-specific voting state
            state = self.get_current_state(mafia_id)
            state += f"\nMafia Kill Vote Phase:\n"
            state += f"Your mafia teammate(s): Players {living_mafia}\n"
            
            # Show current votes
            if self.state.current_mafia_votes:
                state += "\nCurrent votes:\n"
                for voter, target in self.state.current_mafia_votes.items():
                    state += f"Player {voter} voted to kill Player {target}\n"
            
            state += "\nUse 'KILL: player_id' to vote for tonight's target."
            
            state_message = {
                "role": "user",
                "content": state
            }
            
            # Get and process vote
            response = current_player.get_response(state_message, self.get_game_image())
            
            vote_match = self.kill_pattern.match(response)
            if vote_match:
                target = int(vote_match.group(1))
                if target in self.state.living_players and target not in living_mafia:
                    self.state.current_mafia_votes[mafia_id] = target
                    print(f"Player {mafia_id} voted to kill Player {target}")
            else:
                print(f"Player {mafia_id} did not make a valid KILL vote. Response was: {response}")
        
        print("\nResults:")
        print("-" * 20)
        
        print("\nVotes cast:")
        for mafia_id, target in self.state.current_mafia_votes.items():
            print(f"Player {mafia_id} voted to kill Player {target}")
        
        print("\nFinal decision:")
        if len(self.state.current_mafia_votes) == len(living_mafia):
            votes = list(self.state.current_mafia_votes.values())
            if len(set(votes)) == 1:
                self.state.mafia_kill_target = votes[0]
                print(f"Mafia team chose to kill Player {self.state.mafia_kill_target}")
            else:
                print("Mafia team could not agree on a target. No kill tonight.")
        else:
            print("Not all mafia members voted. No kill tonight.")