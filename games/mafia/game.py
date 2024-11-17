# games/mafia/game.py

from enum import Enum
from typing import Dict, Optional, List, Set, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import re
import base64
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
import io
import random

from base.llm_player import BaseLLMPlayer

class GamePhase(Enum):
    SETUP = "setup"
    NIGHT = "night"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTING = "day_voting"
    GAME_END = "game_end"

class PlayerRole(Enum):
    VILLAGER = "villager"
    MAFIA = "mafia"
    DETECTIVE = "detective"
    DOCTOR = "doctor"

@dataclass
class PlayerState:
    """Tracks individual player state and information"""
    role: PlayerRole
    is_alive: bool = True
    current_action: Optional[str] = None
    action_history: List[str] = field(default_factory=list)
    vote_history: List[int] = field(default_factory=list)
    night_action_used: bool = False
    was_saved: bool = False
    was_investigated: bool = False
    known_roles: Dict[int, PlayerRole] = field(default_factory=dict)
    suspicious_of: List[int] = field(default_factory=list)
    received_votes: int = 0
    private_messages: List[str] = field(default_factory=list)
    team_messages: List[str] = field(default_factory=list)
    last_action_result: Optional[str] = None
    discussion_contributions: List[str] = field(default_factory=list)

@dataclass
class GameState:
    """Tracks overall game state"""
    current_phase: GamePhase = GamePhase.SETUP
    players: Dict[int, PlayerState] = field(default_factory=dict)
    living_players: Set[int] = field(default_factory=set)
    dead_players: Set[int] = field(default_factory=set)
    day_count: int = 0
    phase_messages: List[str] = field(default_factory=list)
    action_queue: List[Tuple[int, str]] = field(default_factory=list)
    vote_counts: Dict[int, int] = field(default_factory=dict)
    mafia_votes: Dict[int, int] = field(default_factory=dict)
    protected_player: Optional[int] = None
    investigation_results: Dict[int, PlayerRole] = field(default_factory=dict)
    discussion_round: int = 0
    phase_history: List[str] = field(default_factory=list)
    game_events: List[str] = field(default_factory=list)
    win_condition: Optional[str] = None
    rounds_without_kill: int = 0
    mafia_coordination_attempts: int = 0
    mafia_team_chat: List[str] = field(default_factory=list)
    current_mafia_votes: Dict[int, int] = field(default_factory=dict)

@dataclass
class MafiaConfig:
    """Configuration for Mafia game"""
    run_dir: Path
    max_turns: int = 100
    roles: Dict[PlayerRole, int] = field(default_factory=lambda: {
        PlayerRole.MAFIA: 2,
        PlayerRole.DETECTIVE: 1,
        PlayerRole.DOCTOR: 1,
        PlayerRole.VILLAGER: 3
    })
    max_discussion_rounds: int = 3
    night_action_timeout: int = 30
    discussion_timeout: int = 60
    voting_timeout: int = 30
    max_rounds_without_kill: int = 5

class MafiaGame:
    def __init__(self, run_dir: Path, max_turns: int = 100):
        """Initialize the Mafia game"""
        self.config = MafiaConfig(
            run_dir=run_dir,
            max_turns=max_turns
        )
        self.state = GameState()
        
        # Action validation patterns
        self.kill_pattern = re.compile(r"KILL:\s*(\d+)")
        self.save_pattern = re.compile(r"SAVE:\s*(\d+)")
        self.investigate_pattern = re.compile(r"INVESTIGATE:\s*(\d+)")
        self.vote_pattern = re.compile(r"VOTE:\s*(\d+)")
        self.say_pattern = re.compile(r"SAY:\s*(.+)")
        self.team_pattern = re.compile(r"TEAM:\s*(.+)")
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None

        
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are playing Mafia, a game of social deduction. There are 7 players:
                2 Mafia, 1 Detective, 1 Doctor, and 3 Villagers.
                
                Use these exact formats for actions:
                - Mafia kill: 'KILL: player_id'
                - Doctor save: 'SAVE: player_id'
                - Detective investigate: 'INVESTIGATE: player_id'
                - Day discussion: 'SAY: your_message'
                - Voting: 'VOTE: player_id' or 'VOTE: 0' to abstain if unsure
                - Mafia team chat: 'TEAM: your_message'
                
                Game Flow:
                1. Night Phase: Special roles take actions
                2. Day Discussion: All living players discuss
                3. Day Voting: All living players vote to eliminate (can abstain by voting 0)
                
                Win Conditions:
                - Town wins when all Mafia are eliminated
                - Mafia wins when they equal or outnumber Town
                
                Use deduction to identify players to try to win the game."""
        }

    def initialize_roles(self, players: List[BaseLLMPlayer]) -> None:
        """Randomly assign roles to players"""
        all_roles = []
        for role, count in self.config.roles.items():
            all_roles.extend([role] * count)
        random.shuffle(all_roles)
        
        mafia_members = []
        print("\n=== GAME SETUP ===")
        print("Role Assignments:")
        
        for idx, role in enumerate(all_roles):
            player_id = idx + 1
            self.state.players[player_id] = PlayerState(role=role)
            self.state.living_players.add(player_id)
            print(f"Player {player_id}: {role.value}")
            
            if role == PlayerRole.MAFIA:
                mafia_members.append(player_id)
        
        print(f"\nMafia Team: Players {mafia_members}")
        print("=================\n")
                
        # Share mafia member information
        for member_id in mafia_members:
            team_info = f"You are Mafia. Your teammate is Player {[m for m in mafia_members if m != member_id][0]}"
            self.state.players[member_id].team_messages.append(team_info)
            
    def get_current_state(self, player_id: Optional[int] = None) -> str:
        base_state = (
            f"Day {self.state.day_count}\n"
            f"Phase: {self.state.current_phase.value}\n"
            f"Living players: {sorted(self.state.living_players)}\n"
            f"Dead players: {sorted(self.state.dead_players)}\n\n"
        )

        if player_id is None:
            return base_state

        # Add clear self-identification
        base_state += f"You are Player {player_id}\n"
        
        player_state = self.state.players[player_id]
        base_state += f"Your role: {player_state.role.value}\n"
        
        # Add team information for Mafia
        if player_state.role == PlayerRole.MAFIA:
            # Find Mafia teammate
            mafia_teammates = [
                pid for pid, p in self.state.players.items()
                if p.role == PlayerRole.MAFIA and pid != player_id
            ]
            
            for teammate_id in mafia_teammates:
                teammate_status = "alive" if teammate_id in self.state.living_players else "dead"
                base_state += f"Your Mafia teammate is Player {teammate_id} ({teammate_status})\n"
                
            if self.state.current_phase == GamePhase.NIGHT:
                if player_id in self.state.mafia_votes:
                    base_state += f"You voted to kill Player {self.state.mafia_votes[player_id]}\n"
                    for teammate_id in mafia_teammates:
                        if teammate_id in self.state.mafia_votes:
                            base_state += f"Your teammate voted to kill Player {self.state.mafia_votes[teammate_id]}\n"

        # Add role-specific information
        elif player_state.role == PlayerRole.DETECTIVE:
            investigation_results = []
            for target, role in self.state.investigation_results.items():
                investigation_results.append(
                    f"Player {target}: {'Mafia' if role == PlayerRole.MAFIA else 'Town'}"
                )
            if investigation_results:
                base_state += "Investigation results:\n" + "\n".join(investigation_results) + "\n"

        elif player_state.role == PlayerRole.DOCTOR:
            if player_id == self.state.protected_player:
                base_state += "You protected yourself last night.\n"
            elif self.state.protected_player:
                base_state += f"You protected Player {self.state.protected_player} last night.\n"

        # Add discussion history during day phase
        if self.state.current_phase == GamePhase.DAY_DISCUSSION:
            recent_messages = self.state.phase_messages[-10:]  # Show last 10 messages
            if recent_messages:
                base_state += "\nRecent Discussion:\n" + "\n".join(recent_messages) + "\n"

        action_prompt = self._get_action_prompt(player_id)
        return base_state + "\n" + action_prompt

    def _get_action_prompt(self, player_id: int) -> str:
        """Get the appropriate action prompt with clear player identification"""
        player = self.state.players[player_id]
        
        if not player.is_alive:
            return f"You (Player {player_id}) are dead. Watch the game proceed."
            
        if self.state.current_phase == GamePhase.NIGHT:
            if player.night_action_used:
                return f"You (Player {player_id}) have already used your night action."
                
            prompts = {
                PlayerRole.MAFIA: f"Submit your kill target with 'KILL: player_id' (you cannot target yourself or your teammate)",
                PlayerRole.DOCTOR: f"Choose a player to protect with 'SAVE: player_id'",
                PlayerRole.DETECTIVE: f"Investigate a player with 'INVESTIGATE: player_id' (you cannot target yourself)",
                PlayerRole.VILLAGER: f"You (Player {player_id}) are a Villager. Wait for night phase to end."
            }
            return prompts[player.role]
            
        elif self.state.current_phase == GamePhase.DAY_DISCUSSION:
            return f"You (Player {player_id}) can participate in discussion using 'SAY: your_message'"
            
        elif self.state.current_phase == GamePhase.DAY_VOTING:
            return f"You (Player {player_id}) must vote for elimination using 'VOTE: player_id' (cannot vote for yourself)"
            
        return ""

    def _process_night_actions(self) -> None:
        """Process night actions in strict order"""
        # Detective Phase
        print("\nDetective Results:")
        for target_id, result in self.state.investigation_results.items():
            print(f"Player {target_id} investigated: {result}")
    
        # Mafia Phase
        print("\nMafia Coordination:")
        print(f"Current votes: {self.state.mafia_votes}")
        print(f"Coordination attempts: {self.state.mafia_coordination_attempts}")
        mafia_kill_target = self._handle_mafia_coordination()
        print(f"Final kill target: {mafia_kill_target}")

        # Doctor Phase
        protected_id = None
        for player_id, player in self.state.players.items():
            if player.role == PlayerRole.DOCTOR and player.current_action:
                match = self.save_pattern.match(player.current_action)
                if match:
                    protected_id = int(match.group(1))
                    self.state.protected_player = protected_id

        print("\nDoctor Action:")
        print(f"Protected player: {self.state.protected_player}")

        # Resolve Kill Attempt
        if mafia_kill_target:
            if mafia_kill_target == protected_id:
                save_message = "The Doctor successfully saved their target! Your kill attempt failed."
                self.state.game_events.append("Someone was targeted but saved by the Doctor!")
                # Notify mafia members about the failed kill
                for player_id, player in self.state.players.items():
                    if player.role == PlayerRole.MAFIA:
                        player.team_messages.append(save_message)
                self.state.rounds_without_kill += 1
            else:
                self.state.players[mafia_kill_target].is_alive = False
                self.state.living_players.remove(mafia_kill_target)
                self.state.dead_players.add(mafia_kill_target)
                self.state.game_events.append(f"Player {mafia_kill_target} was killed in the night!")
                self.state.rounds_without_kill = 0
        else:
            self.state.game_events.append("The night passes quietly. No one was killed.")
            self.state.rounds_without_kill += 1

        print("\nNight Results:")
        for event in self.state.game_events[-3:]:
            print(event)

    def _handle_mafia_coordination(self) -> Optional[int]:
        """Handle mafia team coordination for kill target"""
        living_mafia = [pid for pid in self.state.living_players 
                        if self.state.players[pid].role == PlayerRole.MAFIA]
        
        if len(living_mafia) != 2:
            # If only one mafia, they can act alone
            if len(living_mafia) == 1 and living_mafia[0] in self.state.mafia_votes:
                return self.state.mafia_votes[living_mafia[0]]
            return None

        print(f"\nMafia Team ({living_mafia})")
        print(f"Coordination Attempt {self.state.mafia_coordination_attempts + 1}/3:")
        
        # Get current votes
        vote1 = self.state.mafia_votes.get(living_mafia[0])
        vote2 = self.state.mafia_votes.get(living_mafia[1])

        if vote1 is None or vote2 is None:
            # Still waiting for votes
            return None

        if vote1 == vote2:
            # Success! Both agree
            coordination_message = f"Mafia team agreed on target: Player {vote1}"
            print(f"Success! {coordination_message}")
            for mafia_id in living_mafia:
                self.state.players[mafia_id].team_messages.append(coordination_message)
            return vote1

        # Disagreement - initiate team chat
        self.state.mafia_coordination_attempts += 1
        print(f"Disagreement: Player {living_mafia[0]} voted {vote1}, Player {living_mafia[1]} voted {vote2}")
        
        if self.state.mafia_coordination_attempts >= 3:
            fail_message = "No agreement reached after 3 attempts. No kill tonight."
            print(fail_message)
            for mafia_id in living_mafia:
                self.state.players[mafia_id].team_messages.append(fail_message)
            return None

        # Clear current votes and notify team
        disagree_message = (
            f"Vote mismatch (Attempt {self.state.mafia_coordination_attempts}/3): "
            f"Player {living_mafia[0]} voted for {vote1}, "
            f"Player {living_mafia[1]} voted for {vote2}. "
            "Discuss and try again."
        )
        
        self.state.mafia_votes.clear()
        for mafia_id in living_mafia:
            self.state.players[mafia_id].team_messages.append(disagree_message)
            
            # Send new state message to force revote
            self.state.players[mafia_id].night_action_used = False
            state_message = self.get_current_state(mafia_id)
            self.state.players[mafia_id].private_messages.append({
                "role": "user",
                "content": state_message + "\n" + disagree_message
            })

        return None

    def _get_mafia_target(self) -> Optional[int]:
        """Determine final mafia target from votes"""
        if not self.state.mafia_votes:
            return None
            
        # Get all mafia members
        mafia_members = [pid for pid, p in self.state.players.items() 
                        if p.role == PlayerRole.MAFIA and p.is_alive]
                        
        # If all living mafia voted and agree
        if len(self.state.mafia_votes) == len(mafia_members):
            votes = list(self.state.mafia_votes.values())
            if all(v == votes[0] for v in votes):
                return votes[0]
                
        return None

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
        
    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process any player action based on current phase"""
        # print(f"Player {player_id} move: {response}")
        
        if not self.state.players[player_id].is_alive:
            return {
                "valid": False,
                "message": "You are dead and cannot take actions.",
                "end_turn": True
            }

        if self.state.current_phase == GamePhase.NIGHT:
            return self._handle_night_action(response, player_id)
        elif self.state.current_phase == GamePhase.DAY_DISCUSSION:
            return self._handle_discussion(response, player_id)
        elif self.state.current_phase == GamePhase.DAY_VOTING:
            return self._handle_vote(response, player_id)
            
        return {
            "valid": False,
            "message": "Invalid game phase for actions.",
            "end_turn": True
        }

    def _handle_night_action(self, response: str, player_id: int) -> Dict:
        """Handle night phase actions with improved Mafia coordination"""
        player = self.state.players[player_id]
        
        if player.night_action_used:
            return {
                "valid": False,
                "message": "You have already used your night action.",
                "end_turn": True
            }

        if player.role == PlayerRole.MAFIA:
            match = self.kill_pattern.match(response)
            if not match:
                return {
                    "valid": False,
                    "message": "Invalid kill format. Use 'KILL: player_id'",
                    "end_turn": False
                }
            
            target_id = int(match.group(1))
            if not self._validate_target(target_id, player_id):
                return {
                    "valid": False,
                    "message": "Invalid target selection.",
                    "end_turn": False
                }

            self.state.mafia_votes[player_id] = target_id
            player.night_action_used = True

            # Notify other mafia member if they're alive
            other_mafia = [pid for pid in self.state.living_players 
                        if self.state.players[pid].role == PlayerRole.MAFIA 
                        and pid != player_id]
            
            if other_mafia:
                self.state.players[other_mafia[0]].team_messages.append(
                    f"Your teammate (Player {player_id}) voted to kill Player {target_id}"
                )

        elif player.role == PlayerRole.DOCTOR:
            match = self.save_pattern.match(response)
            if not match:
                return {
                    "valid": False,
                    "message": "Invalid save format. Use 'SAVE: player_id'",
                    "end_turn": False
                }
            target_id = int(match.group(1))
            if not self._validate_target(target_id, player_id):
                return {
                    "valid": False,
                    "message": "Invalid target selection.",
                    "end_turn": False
                }
            self.state.protected_player = target_id
            player.night_action_used = True

        elif player.role == PlayerRole.DETECTIVE:
            match = self.investigate_pattern.match(response)
            if not match:
                return {
                    "valid": False,
                    "message": "Invalid investigate format. Use 'INVESTIGATE: player_id'",
                    "end_turn": False
                }
            target_id = int(match.group(1))
            if not self._validate_target(target_id, player_id):
                return {
                    "valid": False,
                    "message": "Invalid target selection.",
                    "end_turn": False
                }
            self.state.investigation_results[target_id] = self.state.players[target_id].role
            player.night_action_used = True

        return {
            "valid": True,
            "message": "Action submitted successfully.",
            "end_turn": True
        }

    def _handle_discussion(self, response: str, player_id: int) -> Dict:
        """Process day discussion messages"""
        # Only accept SAY commands during discussion
        match = self.say_pattern.match(response)
        if not match:
            return {
                "valid": False,
                "message": "Invalid discussion format. Use 'SAY: your_message'",
                "end_turn": False
            }
        
        message = match.group(1)
        formatted_message = f"Player {player_id} says: {message}"
        self.state.phase_messages.append(formatted_message)
        self.state.players[player_id].discussion_contributions.append(message)

        print(f"\nDiscussion - Player {player_id} says: {response}")

        return {
            "valid": True,
            "message": "Message shared with all players.",
            "end_turn": True
        }


    def _handle_vote(self, response: str, player_id: int) -> Dict:
        """Process voting phase actions"""
        match = self.vote_pattern.match(response)
        if not match:
            return {
                "valid": False,
                "message": "Invalid vote format. Use 'VOTE: player_id' or 'VOTE: 0' to abstain",
                "end_turn": False
            }
            
        target_id = int(match.group(1))
        
        # Handle abstain vote
        if target_id == 0:
            vote_message = f"Player {player_id} has abstained from voting."
            self.state.phase_messages.append(vote_message)
            self.state.players[player_id].vote_history.append(0)
            return {
                "valid": True,
                "message": "Abstain vote recorded.",
                "end_turn": True
            }
        
        # Validate vote target
        if target_id not in self.state.living_players:
            return {
                "valid": False,
                "message": "Cannot vote for dead players.",
                "end_turn": False
            }
            
        if target_id == player_id:
            return {
                "valid": False,
                "message": "Cannot vote for yourself.",
                "end_turn": False
            }
        
        # Record the vote
        self.state.vote_counts[target_id] = self.state.vote_counts.get(target_id, 0) + 1
        self.state.players[player_id].vote_history.append(target_id)
        
        vote_message = f"Player {player_id} has voted."
        self.state.phase_messages.append(vote_message)
        
        return {
            "valid": True,
            "message": "Vote recorded.",
            "end_turn": True
        }
    
    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Main game loop with proper ordering"""
        self.initialize_roles(players)
        turn = 0
        
        while turn < self.config.max_turns:
            # Night Phase with strict ordering
            self.state.current_phase = GamePhase.NIGHT
            self._reset_night_actions()
            
            # 1. Detective Phase
            for player_id in self.state.living_players:
                player = self.state.players[player_id]
                if player.role == PlayerRole.DETECTIVE:
                    current_player = players[player_id - 1]
                    state_message = {
                        "role": "user",
                        "content": self.get_current_state(player_id)
                    }
                    response = current_player.get_response(state_message, self.get_game_image())
                    self._handle_night_action(response, player_id)

            # 2. Mafia Phase
            living_mafia = [pid for pid in self.state.living_players 
                        if self.state.players[pid].role == PlayerRole.MAFIA]
            for player_id in living_mafia:
                current_player = players[player_id - 1]
                state_message = {
                    "role": "user",
                    "content": self.get_current_state(player_id)
                }
                response = current_player.get_response(state_message, self.get_game_image())
                self._handle_night_action(response, player_id)

            # 3. Doctor Phase
            for player_id in self.state.living_players:
                player = self.state.players[player_id]
                if player.role == PlayerRole.DOCTOR:
                    current_player = players[player_id - 1]
                    state_message = {
                        "role": "user",
                        "content": self.get_current_state(player_id)
                    }
                    response = current_player.get_response(state_message, self.get_game_image())
                    self._handle_night_action(response, player_id)

            # Process night actions in correct order
            self._process_night_actions()
            
            if self._check_win_condition():
                return self.get_game_result()

            # Day Discussion Phase with proper order
            self.state.current_phase = GamePhase.DAY_DISCUSSION
            self.state.day_count += 1
            
            # Announce night results
            night_results = self._get_night_results_announcement()
            for player_id in self.state.living_players:
                self.state.players[player_id].private_messages.append({
                    "role": "user",
                    "content": f"Day {self.state.day_count} begins.\n{night_results}"
                })

            # Five discussion rounds, ordered by player number
            for discussion_round in range(5):
                round_announcement = f"Discussion Round {discussion_round + 1}/3 begins."
                for player_id in sorted(self.state.living_players):  # Maintain player order
                    self.state.players[player_id].private_messages.append({
                        "role": "user",
                        "content": round_announcement
                    })

                    current_player = players[player_id - 1]
                    state_message = {
                        "role": "user",
                        "content": self.get_current_state(player_id)
                    }
                    response = current_player.get_response(state_message, self.get_game_image())
                    self._handle_discussion(response, player_id)

                    # Share message with other living players
                    for listener_id in self.state.living_players:
                        if listener_id != player_id:
                            self.state.players[listener_id].private_messages.append({
                                "role": "user",
                                "content": self.state.phase_messages[-1]
                            })

            # Voting Phase
            self.state.current_phase = GamePhase.DAY_VOTING
            self._reset_votes()
            
            # Announce start of voting
            for player_id in self.state.living_players:
                self.state.players[player_id].private_messages.append({
                    "role": "user",
                    "content": "Discussion has ended. Voting phase begins now. Submit your vote for elimination."
                })
            
            # Collect one vote from each living player
            vote_order = list(self.state.living_players)
            random.shuffle(vote_order)
            
            for voter_id in vote_order:
                current_player = players[voter_id - 1]
                state_message = {
                    "role": "user",
                    "content": self.get_current_state(voter_id)
                }
                response = current_player.get_response(state_message, self.get_game_image())
                self._handle_vote(response, voter_id)

            # Process elimination
            self._process_elimination()
            
            # Announce voting results to all living players
            vote_results = self._get_vote_results_announcement()
            for player_id in self.state.living_players:
                self.state.players[player_id].private_messages.append({
                    "role": "user",
                    "content": vote_results
                })
            
            if self._check_win_condition():
                return self.get_game_result()
                
            turn += 1
        
        return self.get_game_result()
    
    def _get_night_results_announcement(self) -> str:
        """Format night phase results announcement"""
        if not self.state.game_events:
            return "The night passes quietly."
        return "\n".join(self.state.game_events[-3:])  # Last 3 events

    def _get_vote_results_announcement(self) -> str:
        """Format voting results announcement"""
        results = ["Voting Results:"]
        for player_id, votes in self.state.vote_counts.items():
            results.append(f"Player {player_id}: {votes} votes")
        
        if self.state.players[max(self.state.vote_counts, key=self.state.vote_counts.get)].role == PlayerRole.MAFIA:
            results.append("\nA member of the Mafia was eliminated!")
        else:
            results.append("\nA member of the Town was eliminated!")
            
        return "\n".join(results)

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
            
        if self.state.rounds_without_kill >= self.config.max_rounds_without_kill:
            self.state.win_condition = "Game drawn due to stalemate."
            return True
            
        return False

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
        for player in self.state.players.values():
            player.night_action_used = False
            player.was_saved = False
            player.was_investigated = False
            
        self.state.mafia_votes.clear()
        self.state.protected_player = None

    def _reset_votes(self) -> None:
        """Reset voting trackers"""
        self.state.vote_counts.clear()
        for player in self.state.players.values():
            player.received_votes = 0

    def _process_elimination(self) -> None:
        """Process voting results and eliminate player if clear majority"""
        if not self.state.vote_counts:
            self.state.game_events.append(
                "No votes were cast or all players abstained. No one was eliminated."
            )
            return

        # Find player(s) with most votes
        max_votes = max(self.state.vote_counts.values())
        most_voted = [pid for pid, votes in self.state.vote_counts.items() 
                    if votes == max_votes]

        # Calculate total votes cast (excluding abstentions)
        total_votes = sum(self.state.vote_counts.values())
        living_players = len(self.state.living_players)
        abstained = living_players - total_votes

        # Only eliminate if there's a single player with the most votes
        if len(most_voted) == 1 and max_votes > abstained:
            player_id = most_voted[0]
            self.state.players[player_id].is_alive = False
            self.state.living_players.remove(player_id)
            self.state.dead_players.add(player_id)
            
            role_reveal = "Mafia" if self.state.players[player_id].role == PlayerRole.MAFIA else "Town"
            self.state.game_events.append(
                f"Player {player_id} was eliminated by town vote!\n"
                f"They were a member of the {role_reveal}!"
            )
        else:
            self.state.game_events.append(
                f"No elimination occurred. {abstained} players abstained from voting."
            )

        print("\n=== VOTING RESULTS ===")
        print(f"Vote counts: {self.state.vote_counts}")
        print(f"Abstained: {abstained}")
        for event in self.state.game_events[-3:]:
            print(event)

    def get_game_result(self) -> Dict[str, str]:
        """Get the final game result"""
        self.end_time = datetime.now().isoformat()
        
        living_mafia = [pid for pid, p in self.state.players.items()
                        if p.role == PlayerRole.MAFIA and p.is_alive]
        living_town = [pid for pid, p in self.state.players.items()
                        if p.role != PlayerRole.MAFIA and p.is_alive]
        
        return {
            "status": "complete",
            "winner": self.state.win_condition,
            "days_elapsed": self.state.day_count,
            "living_mafia": living_mafia,
            "living_town": living_town,
            "game_events": self.state.game_events,
            "final_state": self.get_current_state()
        }

    def get_final_position(self) -> str:
        """Get string representation of final game state"""
        final_state = [
            f"Game ended after {self.state.day_count} days",
            f"Result: {self.state.win_condition}",
            "\nFinal role reveal:"
        ]
        
        for player_id, player in self.state.players.items():
            status = "alive" if player.is_alive else "dead"
            final_state.append(
                f"Player {player_id}: {player.role.value} ({status})"
            )
            
        final_state.append("\nGame events:")
        final_state.extend(self.state.game_events)
        
        return "\n".join(final_state)