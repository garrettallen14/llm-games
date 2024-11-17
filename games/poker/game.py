from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import base64
import numpy as np
from pettingzoo.classic import texas_holdem_v4
from base.llm_player import BaseLLMPlayer
import pygame
import os
import re

@dataclass
class TournamentState:
    """Tracks tournament progress"""
    player_stacks: Dict[int, int] = field(default_factory=dict)
    eliminated_players: List[int] = field(default_factory=list)
    current_level: int = 0
    hands_played: int = 0
    start_time: datetime = field(default_factory=datetime.now)

@dataclass
class TournamentConfig:
    """Tournament settings"""
    starting_stack: int = 10000
    blind_schedule: List[tuple] = field(default_factory=lambda: [
        (50, 100), (100, 200), (150, 300), (200, 400),
        (300, 600), (400, 800), (500, 1000), (600, 1200)
    ])
    hands_per_level: int = 10
    min_players: int = 2

class PokerGame:
    def __init__(self, run_dir: Path, max_turns: int = 1000):
        """Initialize tournament structure"""
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.config = TournamentConfig()
        self.state = None
        self.env = None
        self.move_pattern = re.compile(r"MOVE:\s*(CALL|FOLD|RAISE\s+\d+|CHECK)")
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the initial system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are playing in a Texas Hold'em Poker tournament.

Your responses must be EXACTLY in one of these formats:
MOVE: CALL   - to match the current bet
MOVE: FOLD   - to fold your hand
MOVE: CHECK  - when there's no bet to call
MOVE: RAISE X - to raise to amount X

For example:
MOVE: CALL
MOVE: RAISE 300
MOVE: FOLD
MOVE: CHECK

Do not add any explanation - just respond with the exact move format.
Always check the legal actions available to you and only make legal moves.
Consider pot odds, position, and tournament stage when making decisions.
Play to maximize your chance of winning the tournament."""
        }

    def setup_tournament(self, player_ids: List[int]):
        """Initialize tournament state"""
        if len(player_ids) < self.config.min_players:
            raise ValueError(f"Need at least {self.config.min_players} players")
            
        self.state = TournamentState(
            player_stacks={pid: self.config.starting_stack for pid in player_ids}
        )
        
        # Create our own Texas Hold'em environment that supports more players
        self.num_players = len(player_ids)
        self.env = texas_holdem_v4.env(num_players=self.num_players, render_mode="rgb_array")
        self.env.reset()

    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get current game state from player's perspective"""
        if not player_id:
            return self._get_public_state()
        
        agent_name = f'player_{player_id-1}'
        obs = self.env.observe(agent_name)
        
        if obs is None:
            return self._get_public_state()

        # Extract observation and action mask
        observation = obs['observation']
        action_mask = obs['action_mask']
        
        # Calculate current bet and pot
        betting_info = observation[52:72]
        current_bet = int(max(betting_info)) if betting_info.any() else 0
        total_pot = int(sum(betting_info))
        
        # Calculate minimum raise
        min_raise = max(
            current_bet * 2,  # Double current bet
            self.config.blind_schedule[self.state.current_level][1] * 2  # Or double big blind
        )
        
        # Build state description
        state_parts = [
            f"\nTournament Level: {self.state.current_level + 1}",
            f"Blinds: {self.config.blind_schedule[self.state.current_level][0]}/{self.config.blind_schedule[self.state.current_level][1]}",
            f"Your Stack: {self.state.player_stacks[player_id]}",
            f"Players Remaining: {len(self.state.player_stacks) - len(self.state.eliminated_players)}"
        ]

        # Add stack sizes for other players
        state_parts.append("\nStack Sizes:")
        for pid, stack in self.state.player_stacks.items():
            if pid != player_id and pid not in self.state.eliminated_players:
                state_parts.append(f"Player {pid}: {stack}")

        # Get cards - only show this player's hole cards
        visible_cards = self._get_visible_cards(observation[:52])
        hand_cards = visible_cards[:2]  # First two are hole cards
        community_cards = visible_cards[2:]  # Rest are community cards

        state_parts.extend([
            f"\nYour Hand: {' '.join(hand_cards)}",
            f"Community Cards: {' '.join(community_cards)}",
            f"Pot Size: {total_pot}",
            f"Current Bet: {current_bet}"
        ])

        # Add legal actions with proper amounts
        legal_actions = []
        if action_mask[0]: legal_actions.append(f"CALL ({current_bet})")
        if action_mask[1]: legal_actions.append(f"RAISE (min: {min_raise})")
        if action_mask[2]: legal_actions.append("FOLD")
        if action_mask[3]: legal_actions.append("CHECK")

        state_parts.append(f"\nLegal Actions: {', '.join(legal_actions)}")
        
        # Print nicely formatted state to terminal
        print("\n" + "="*50)
        print(f"Player {player_id}'s Turn")
        print("="*50)
        print("\n".join(state_parts))
        print("-"*50)
        
        return "\n".join(state_parts)

    def _get_visible_cards(self, card_obs: np.ndarray) -> List[str]:
        """Convert card observation to list of card strings"""
        suits = "♠♥♦♣"
        ranks = "A23456789TJQK"
        cards = []
        
        for i in range(52):
            if card_obs[i] == 1:
                suit_idx = i // 13
                rank_idx = i % 13
                cards.append(f"{suits[suit_idx]}{ranks[rank_idx]}")
        
        return cards

    def _get_public_state(self) -> str:
        """Get public tournament state"""
        active_players = len(self.state.player_stacks) - len(self.state.eliminated_players)
        current_blinds = self.config.blind_schedule[self.state.current_level]
        
        return f"""Tournament Status:
Level: {self.state.current_level + 1}
Blinds: {current_blinds[0]}/{current_blinds[1]}
Hands Played: {self.state.hands_played}
Players Remaining: {active_players}"""

    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        # Get current observation and action mask
        agent_name = f'player_{player_id-1}'
        obs = self.env.observe(agent_name)
        
        if obs is None:
            return {
                "valid": False,
                "message": "Not your turn to act",
                "end_turn": True,
                "end_game": False,
                "skip_inference": True
            }

        action_mask = obs['action_mask']
        
        # Parse the move
        match = self.move_pattern.search(response)
        if not match:
            return {
                "valid": False,
                "message": "Invalid move format. Use MOVE: ACTION",
                "end_turn": False
            }

        # Convert move to action number and validate
        try:
            action = self._convert_move_to_action(match.group(1))
            
            # Check if action is legal
            if not action_mask[action]:
                return {
                    "valid": False,
                    "message": f"Illegal action. Legal actions: {self._get_legal_actions(action_mask)}",
                    "end_turn": False
                }
                
            # Execute the action
            self.env.step(action)
            
            # Check if hand is complete
            _, reward, terminated, truncated, _ = self.env.last()
            
            if terminated or truncated:
                self._update_stacks()
                self._check_eliminations()
                self._update_tournament_state()
            
            return {
                "valid": True,
                "message": f"Move accepted: {match.group(1)}",
                "end_turn": True,
                "end_game": self._is_tournament_complete(),
                "skip_inference": False
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error processing move: {str(e)}",
                "end_turn": False
            }

    def _convert_move_to_action(self, move_str: str) -> int:
        """Convert move string to PettingZoo action number"""
        if move_str == "CALL": return 0
        if move_str.startswith("RAISE"): return 1
        if move_str == "FOLD": return 2
        if move_str == "CHECK": return 3
        raise ValueError(f"Invalid move: {move_str}")

    def _get_legal_actions(self, action_mask) -> List[str]:
        """Get list of legal actions from mask"""
        actions = []
        if action_mask[0]: actions.append("CALL")
        if action_mask[1]: actions.append("RAISE")
        if action_mask[2]: actions.append("FOLD")
        if action_mask[3]: actions.append("CHECK")
        return actions

    def _update_stacks(self):
        """Update player stacks after hand completion"""
        for pid in self.state.player_stacks:
            if pid not in self.state.eliminated_players:
                agent_name = f'player_{pid-1}'
                obs = self.env.observe(agent_name)
                if obs is not None:
                    # Get raw chips from the observation
                    raw_obs = obs.get('info', {}).get('raw_obs', {})
                    if 'my_chips' in raw_obs:
                        self.state.player_stacks[pid] = int(raw_obs['my_chips'])

    def _check_eliminations(self):
        """Check for eliminated players"""
        for pid, stack in self.state.player_stacks.items():
            if stack <= 0 and pid not in self.state.eliminated_players:
                self.state.eliminated_players.append(pid)

    def _update_tournament_state(self):
        """Update tournament state after each hand"""
        self.state.hands_played += 1
        
        # Check for blind level increase
        if self.state.hands_played % self.config.hands_per_level == 0:
            self.state.current_level = min(
                self.state.current_level + 1,
                len(self.config.blind_schedule) - 1
            )

    def _is_tournament_complete(self) -> bool:
        """Check if tournament is complete"""
        active_players = len(self.state.player_stacks) - len(self.state.eliminated_players)
        return active_players <= 1

    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the complete tournament"""
        print("\n" + "="*50)
        print("Starting Poker Tournament")
        print(f"Players: {len(players)}")
        print(f"Starting Stack: {self.config.starting_stack}")
        print("="*50 + "\n")
        
        self.setup_tournament([p.player_id for p in players])
        
        hand_count = 0
        while True:
            if self._is_tournament_complete() or hand_count >= self.max_turns:
                break
                
            try:
                print("\n" + "="*50)
                print(f"Hand #{hand_count + 1}")
                print(f"Blinds: {self.config.blind_schedule[self.state.current_level][0]}/"
                      f"{self.config.blind_schedule[self.state.current_level][1]}")
                print("Current Stacks:")
                for pid, stack in self.state.player_stacks.items():
                    if pid not in self.state.eliminated_players:
                        print(f"Player {pid}: {stack}")
                print("="*50 + "\n")
                
                # Reset environment for new hand
                self.env.reset()
                current_hand_reward = {pid: 0 for pid in self.state.player_stacks}
                
                # Try to save initial board state
                print("Saving initial board state for hand...")
                self.get_game_image()
                
                # Play hand
                for agent in self.env.agent_iter():
                    player_id = int(agent.split('_')[1]) + 1
                    
                    # Skip eliminated players
                    if player_id in self.state.eliminated_players:
                        self.env.step(2)  # Auto-fold for eliminated players
                        continue
                    
                    observation, reward, terminated, truncated, info = self.env.last()
                    
                    if terminated or truncated:
                        current_hand_reward[player_id] = reward
                        action = None
                    else:
                        print(f"\nSaving board state for Player {player_id}'s turn...")
                        # Generate and save the current board state image
                        game_image = self.get_game_image()  # This will save to board.png
                        
                        # Get player action
                        state_message = {
                            "role": "user",
                            "content": self.get_current_state(player_id)
                        }
                        
                        try:
                            response = players[player_id-1].get_response(state_message, game_image)
                            print(f"Player {player_id} response: {response}")
                            
                            if response:
                                match = self.move_pattern.search(response)
                                if match:
                                    action = self._convert_move_to_action(match.group(1))
                                else:
                                    print(f"Invalid move format from player {player_id}: {response}")
                                    action = 2  # FOLD if invalid format
                            else:
                                print(f"No response from player {player_id}")
                                action = 2  # FOLD if no response
                        except Exception as e:
                            print(f"Error getting response from player {player_id}: {e}")
                            action = 2  # FOLD on error
                    
                    try:
                        self.env.step(action)
                        # Try to save board state after action
                        print(f"Saving board state after Player {player_id}'s action...")
                        self.get_game_image()
                    except Exception as e:
                        print(f"Error executing step for player {player_id}: {e}")
                        break
                
                # Update stacks based on hand results
                print("\nHand Results:")
                for pid, reward in current_hand_reward.items():
                    if pid not in self.state.eliminated_players:
                        old_stack = self.state.player_stacks[pid]
                        new_stack = max(0, old_stack + int(reward))
                        self.state.player_stacks[pid] = new_stack
                        print(f"Player {pid}: {old_stack} -> {new_stack} ({'+' if reward >= 0 else ''}{reward})")
                
                # Process end of hand
                self._check_eliminations()
                if self.state.eliminated_players:
                    print("\nEliminated Players:", self.state.eliminated_players)
                
                self._update_tournament_state()
                hand_count += 1
                
            except Exception as e:
                print(f"Error during hand {hand_count}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        result = self.get_tournament_results()
        
        print("\n" + "="*50)
        print("Tournament Complete!")
        print(f"Winner: Player {result['winner']}")
        print(f"Hands Played: {result['hands_played']}")
        print("Final Stacks:")
        for pid, stack in result['final_stacks'].items():
            print(f"Player {pid}: {stack}")
        print("="*50 + "\n")
        
        return result

    def get_tournament_results(self) -> Dict:
        """Get final tournament results"""
        active_players = [pid for pid in self.state.player_stacks 
                         if pid not in self.state.eliminated_players]
        
        if len(active_players) > 1:
            # Tournament didn't complete properly
            return {
                "status": "incomplete",
                "active_players": active_players,
                "eliminated_players": self.state.eliminated_players,
                "hands_played": self.state.hands_played,
                "final_stacks": self.state.player_stacks,
                "duration": str(datetime.now() - self.state.start_time)
            }
        
        if not active_players and not self.state.eliminated_players:
            # No players finished the tournament
            return {
                "status": "error",
                "message": "No players completed the tournament",
                "hands_played": self.state.hands_played,
                "duration": str(datetime.now() - self.state.start_time)
            }
        
        # Normal completion - we have a winner
        winner_id = active_players[0] if active_players else self.state.eliminated_players[-1]
        
        # Build complete rankings
        rankings = []
        for i, pid in enumerate(reversed(self.state.eliminated_players[:-1]), start=2):
            rankings.append((pid, i))
        if self.state.eliminated_players:
            rankings.append((self.state.eliminated_players[-1], 1))
        
        return {
            "status": "complete",
            "winner": f"Player {winner_id}",
            "rankings": rankings,
            "hands_played": self.state.hands_played,
            "final_stacks": self.state.player_stacks,
            "duration": str(datetime.now() - self.state.start_time)
        }

    def get_game_image(self, current_player_id: Optional[int] = None) -> Optional[str]:
        """Generate visualization of current game state and save to disk"""
        try:
            print("Attempting to render game state...")
            
            # Calculate optimal dimensions based on number of players and cards
            base_size = 1000
            num_players = len(self.env.possible_agents)
            
            # Calculate rows needed (2 players per row)
            num_rows = (num_players + 1) // 2  # +1 for community cards row
            
            # Calculate dimensions
            screen_height = base_size
            screen_width = int(base_size * 1.5)  # 3:2 aspect ratio
            
            # Calculate card size based on screen dimensions and number of players
            card_height = int(screen_height / (num_rows + 1))  # +1 for spacing
            card_width = int(card_height * 0.7)  # Standard card ratio
            
            pygame.init()
            screen = pygame.Surface((screen_width, screen_height))
            
            # Fill background
            bg_color = (7, 99, 36)
            white = (255, 255, 255)
            screen.fill(bg_color)
            
            def create_card_surface(card, hidden=False):
                surface = pygame.Surface((card_width, card_height))
                if hidden:
                    surface.fill((150, 150, 150))
                else:
                    surface.fill((200, 200, 200))
                pygame.draw.rect(surface, (100, 100, 100), surface.get_rect(), 2)
                
                if not hidden:
                    font = pygame.font.SysFont('Arial', int(card_height/3))
                    text = font.render(card, True, (0, 0, 0))
                    text_rect = text.get_rect(center=surface.get_rect().center)
                    surface.blit(text, text_rect)
                return surface
            
            # Get game state
            base_env = self.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # Draw players in rows
            players_per_row = 2
            for i, player in enumerate(self.env.possible_agents):
                player_id = i + 1
                state = base_env.game.get_state(i)
                
                # Calculate position for this player
                row = i // players_per_row
                col = i % players_per_row
                
                base_x = col * (screen_width // players_per_row) + (screen_width // (players_per_row * 2))
                base_y = (row + 1) * (screen_height // (num_rows + 1))
                
                # Draw player info
                font = pygame.font.SysFont('Arial', 36)
                name_text = f"Player {player_id}"
                if player_id == current_player_id:
                    name_text += " (You)"
                text = font.render(name_text, True, white)
                text_rect = text.get_rect(center=(base_x, base_y - card_height//2))
                screen.blit(text, text_rect)
                
                # Draw stack
                stack_text = font.render(f"Stack: {state.get('my_chips', 0)}", True, white)
                stack_rect = stack_text.get_rect(center=(base_x, base_y - card_height//4))
                screen.blit(stack_text, stack_rect)
                
                # Draw cards
                hand = state.get('hand', [])
                show_cards = (player_id == current_player_id)
                for j, card in enumerate(hand):
                    card_x = base_x + (j - len(hand)/2) * (card_width * 1.2)
                    card_surface = create_card_surface(card, not show_cards)
                    screen.blit(card_surface, (card_x, base_y))
            
            # Draw community cards in center
            community_cards = state.get('public_cards', [])
            if community_cards:
                center_y = screen_height // 2
                for i, card in enumerate(community_cards):
                    card_x = screen_width//2 + (i - len(community_cards)/2) * (card_width * 1.2)
                    card_surface = create_card_surface(card)
                    screen.blit(card_surface, (card_x, center_y))
                
                # Draw pot size
                pot_text = font.render(f"Pot: {state.get('pot', 0)}", True, white)
                pot_rect = pot_text.get_rect(center=(screen_width//2, center_y + card_height * 0.8))
                screen.blit(pot_text, pot_rect)
            
            # Convert and save
            rgb_array = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
            
            if isinstance(rgb_array, np.ndarray):
                import PIL.Image
                img = PIL.Image.fromarray(rgb_array)
                img.save("board.png")
                img.save(str(self.run_dir / "board.png"))
                print(f"Saved board.png successfully")
                
                import io
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            
            return None
            
        except Exception as e:
            print(f"Failed to generate/save game image: {e}")
            import traceback
            traceback.print_exc()
            return None


    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        # Get current observation and action mask
        agent_name = f'player_{player_id-1}'
        obs = self.env.observe(agent_name)
        
        if obs is None:
            return {
                "valid": False,
                "message": "Not your turn to act",
                "end_turn": True,
                "end_game": False,
                "skip_inference": True
            }

        observation = obs['observation']
        action_mask = obs['action_mask']
        
        # Parse the move
        match = self.move_pattern.search(response)
        if not match:
            return {
                "valid": False,
                "message": "Invalid format. Use: MOVE: ACTION",
                "end_turn": False
            }

        move_str = match.group(1)
        
        # Special handling for RAISE
        if move_str.startswith("RAISE"):
            try:
                raise_amount = int(move_str.split()[1])
                current_bet = max(observation[52:72])
                min_raise = current_bet * 2
                
                if raise_amount < min_raise:
                    return {
                        "valid": False,
                        "message": f"Raise must be at least {min_raise}",
                        "end_turn": False
                    }
                
                if raise_amount > self.state.player_stacks[player_id]:
                    return {
                        "valid": False,
                        "message": "Cannot raise more than your stack",
                        "end_turn": False
                    }
            except (IndexError, ValueError):
                return {
                    "valid": False,
                    "message": "Invalid raise format. Use: MOVE: RAISE amount",
                    "end_turn": False
                }

        # Convert move to action number
        try:
            action = self._convert_move_to_action(move_str)
            
            # Validate action against mask
            if not action_mask[action]:
                legal_actions = self._get_legal_actions_with_amounts(observation, action_mask)
                return {
                    "valid": False,
                    "message": f"Illegal action. Legal actions: {', '.join(legal_actions)}",
                    "end_turn": False
                }
                
            # Apply the action
            self.env.step(action)
            
            # Check hand completion
            _, reward, terminated, truncated, _ = self.env.last()
            
            if terminated or truncated:
                self._update_stacks()
                self._check_eliminations()
                self._update_tournament_state()
            
            return {
                "valid": True,
                "message": f"Move accepted: {move_str}",
                "end_turn": True,
                "end_game": self._is_tournament_complete(),
                "skip_inference": False
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error processing move: {str(e)}",
                "end_turn": False
            }

    def _get_legal_actions_with_amounts(self, observation, action_mask) -> List[str]:
        """Get list of legal actions with relevant amounts"""
        actions = []
        current_bet = max(observation[52:72])
        
        if action_mask[0]:  # CALL
            actions.append(f"CALL ({current_bet})")
        if action_mask[1]:  # RAISE
            min_raise = current_bet * 2
            actions.append(f"RAISE (min: {min_raise})")
        if action_mask[2]:  # FOLD
            actions.append("FOLD")
        if action_mask[3]:  # CHECK
            actions.append("CHECK")
        return actions

    def _update_stacks(self):
        """Update player stacks after hand completion"""
        for pid in self.state.player_stacks:
            if pid not in self.state.eliminated_players:
                agent_name = f'player_{pid-1}'
                obs = self.env.observe(agent_name)
                if obs is not None:
                    # Calculate stack from initial stack and rewards
                    _, reward, _, _, info = self.env.last()
                    current_stack = self.state.player_stacks[pid]
                    if isinstance(reward, (int, float)):
                        new_stack = current_stack + int(reward)
                        self.state.player_stacks[pid] = max(0, new_stack)  # Ensure non-negative

    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the complete tournament"""
        print("\n" + "="*50)
        print("Starting Poker Tournament")
        print(f"Players: {len(players)}")
        print(f"Starting Stack: {self.config.starting_stack}")
        print("="*50 + "\n")
        
        self.setup_tournament([p.player_id for p in players])
        
        hand_count = 0
        while True:
            if self._is_tournament_complete() or hand_count >= self.max_turns:
                break
                
            try:
                print("\n" + "="*50)
                print(f"Hand #{hand_count + 1}")
                print(f"Blinds: {self.config.blind_schedule[self.state.current_level][0]}/"
                      f"{self.config.blind_schedule[self.state.current_level][1]}")
                print("Current Stacks:")
                for pid, stack in self.state.player_stacks.items():
                    if pid not in self.state.eliminated_players:
                        print(f"Player {pid}: {stack}")
                print("="*50 + "\n")
                
                # Reset environment for new hand
                self.env.reset()
                current_hand_reward = {pid: 0 for pid in self.state.player_stacks}
                
                # Play hand
                for agent in self.env.agent_iter():
                    player_id = int(agent.split('_')[1]) + 1
                    
                    # Skip eliminated players
                    if player_id in self.state.eliminated_players:
                        self.env.step(2)  # Auto-fold for eliminated players
                        continue
                    
                    observation, reward, terminated, truncated, info = self.env.last()
                    
                    if terminated or truncated:
                        current_hand_reward[player_id] = reward
                        action = None
                    else:
                        print(f"\nPlayer {player_id}'s turn")
                        # Generate game image specific to current player
                        game_image = self.get_game_image(current_player_id=player_id)
                        
                        # Get player action
                        state_message = {
                            "role": "user",
                            "content": self.get_current_state(player_id)
                        }
                        
                        try:
                            response = players[player_id-1].get_response(state_message, game_image)
                            print(f"Player {player_id} response: {response}")
                            
                            if response:
                                match = self.move_pattern.search(response)
                                if match:
                                    action = self._convert_move_to_action(match.group(1))
                                else:
                                    print(f"Invalid move format from player {player_id}: {response}")
                                    action = 2  # FOLD if invalid format
                            else:
                                print(f"No response from player {player_id}")
                                action = 2  # FOLD if no response
                        except Exception as e:
                            print(f"Error getting response from player {player_id}: {e}")
                            action = 2  # FOLD on error
                    
                    try:
                        self.env.step(action)
                    except Exception as e:
                        print(f"Error executing step for player {player_id}: {e}")
                        break
                
                # Update stacks based on hand results
                for pid, reward in current_hand_reward.items():
                    if pid not in self.state.eliminated_players:
                        current_stack = self.state.player_stacks[pid]
                        new_stack = current_stack + int(reward)
                        self.state.player_stacks[pid] = max(0, new_stack)
                
                # Process end of hand
                self._check_eliminations()
                self._update_tournament_state()
                hand_count += 1
                
                print(f"Hand {hand_count} complete. Stacks: {self.state.player_stacks}")
                
            except Exception as e:
                print(f"Error during hand {hand_count}: {str(e)}")
                continue
        
        return self.get_tournament_results()