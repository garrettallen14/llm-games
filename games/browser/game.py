from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
import re
import requests
import json
import html
from dataclasses import dataclass, field

from base.llm_player import BaseLLMPlayer

@dataclass
class BrowserState:
    """Tracks the current state of the 4chan browser"""
    current_view: str = "threads"  # threads, thread
    current_board: str = "biz"
    current_thread: Optional[int] = None
    history: List[Dict] = field(default_factory=list)

class ChanAPI:
    """4chan API wrapper"""
    BASE_URL = "https://a.4cdn.org"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_boards(self) -> Dict:
        """Get list of all boards"""
        response = self.session.get(f"{self.BASE_URL}/boards.json")
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to get boards: {response.status_code}")
    
    def get_catalog(self, board: str) -> List[Dict]:
        """Get catalog of threads for a board"""
        response = self.session.get(f"{self.BASE_URL}/{board}/catalog.json")
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to get catalog: {response.status_code}")
    
    def get_thread(self, board: str, thread_id: int) -> Dict:
        """Get detailed information about a specific thread"""
        response = self.session.get(f"{self.BASE_URL}/{board}/thread/{thread_id}.json")
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to get thread: {response.status_code}")

class BrowserGame:
    """4chan browser game implementation"""
    
    def __init__(self, run_dir: Path, max_turns: int = 20):
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.state = BrowserState()
        self.api = ChanAPI()
        
        # Command patterns
        self.patterns = {
            "read": re.compile(r"READ:\s*(\d+)"),
            "back": re.compile(r"BACK"),
            "favorite": re.compile(r"FAVORITE:\s*([A-Z]+)")
        }
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None
    
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": """YOU'RE A DEGENERATE TRADER TRYING TO FIND THE MOST BASED STOCK TICKER BEING DISCUSSED ON /BIZ/ (BUSINESS & FINANCE) THREADS ON 4CHAN.

OBJECTIVE:
YOUR GOAL IS TO DIVE DEEP INTO /BIZ/ THREADS AND FIND THE MOST BASED STOCK TICKER BEING DISCUSSED. YOU SHOULD:
1. READ THREADS THAT DISCUSS STOCKS AND MARKET OPPORTUNITIES
2. PAY ATTENTION TO SENTIMENT, ANALYSIS, AND DISCUSSION QUALITY
3. WHEN YOU FIND A TICKER THAT YOU BELIEVE HAS THE MOST POTENTIAL, SELECT IT AS YOUR FAVORITE

COMMANDS:
- READ: <thread_id> - READ A SPECIFIC THREAD
- BACK - RETURN TO THREAD LIST
- FAVORITE: <TICKER> - CHOOSE YOUR FAVORITE TICKER (ENDS GAME)

STRATEGY:
1. FIND THREADS THAT ARE GOING WILD OVER A SPECIFIC STOCK
2. FOCUS ON THREADS WITH DEEP DIVE ANALYSIS (CHARTS, NUMBERS, ETC.)
3. CONSIDER BOTH BULLISH AND BEARISH PERSPECTIVES, BUT IF YOU'RE READING THIS YOU'RE PROBABLY A DEGENERATE TRADER ANYWAY
4. WHEN YOU FIND A TICKER THAT'S NOT SHIT AND YOU'RE READY TO RISK IT ALL, USE FAVORITE: TICKER

REMEMBER:
- TAKE TIME TO READ MULTIPLE THREADS

YOUR CURRENT VIEW WILL SHOW YOU AVAILABLE THREADS ON /BIZ/. USE READ TO EXPLORE THEM AND FAVORITE WHEN YOU'VE MADE YOUR CHOICE!!!
YOU MUST VIEW MANY MANY MANYTHREADS BEFORE YOU CAN CHOOSE A FAVORITE TICKER.
ALSO, MAKE ONLY ONE FUCKING MOVE PER TURN. JUST ONE MOVE PER TURN."""
        }
    
    def _clean_html(self, text: str) -> str:
        """Clean HTML from text content"""
        if not text:
            return ""
        # Decode HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def attempt_action(self, response: str) -> Dict:
        """Process player actions with restrictions"""
        if not response:
            return {"valid": False, "message": "No action provided", "end_turn": False}
        
        # Extract the last line that could be a command
        lines = response.strip().split('\n')
        potential_commands = [line.strip().upper() for line in lines]
        
        # Try each line from the end until we find a valid command
        for command in reversed(potential_commands):
            # Check for FAVORITE command first
            favorite_match = self.patterns["favorite"].search(command)
            if favorite_match:
                ticker = favorite_match.group(1)
                return {
                    "valid": True,
                    "message": f"You've chosen {ticker} as your favorite ticker! Game Over.",
                    "end_turn": True,
                    "game_over": True
                }
            
            # Handle other commands
            for cmd_type, pattern in self.patterns.items():
                if match := pattern.search(command):
                    if cmd_type == "read":
                        thread_id = match.group(1)
                        # If we're not in thread list, go back first then read
                        if self.state.current_view != "threads":
                            self.state.current_view = "threads"
                        # Now process the read command
                        return self._read_thread(thread_id)
                    elif cmd_type == "back":
                        if self.state.current_view == "threads":
                            return {"valid": False, "message": "Already viewing /biz/ threads", "end_turn": False}
                        self.state.current_view = "threads"
                        return {"valid": True, "message": "Returned to /biz/ threads", "end_turn": True}
        
        return {
            "valid": False,
            "message": "Invalid command. Use:\nREAD: thread_id\nBACK\nFAVORITE: TICKER",
            "end_turn": False
        }

    def _read_thread(self, thread_id: str) -> Dict:
        """Read a specific thread"""
        try:
            thread_data = self.api.get_thread(self.state.current_board, int(thread_id))
            self.state.current_view = "thread"
            self.state.current_thread = int(thread_id)
            return {
                "valid": True,
                "message": f"Reading thread {thread_id}",
                "end_turn": True
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error reading thread: {str(e)}",
                "end_turn": False
            }
    
    def get_current_state(self) -> str:
        """Get the current view state"""
        try:
            if self.state.current_view == "threads":
                threads_data = self.api.get_catalog(self.state.current_board)
                return f"Current View: /biz/ Threads\n\n" + "\n\n".join(
                    f"Thread {t['no']}: {self._clean_html(t.get('sub', t.get('com', 'No content'))[:100] + '...')}"
                    for page in threads_data
                    for t in page['threads']
                )
            else:
                try:
                    thread_data = self.api.get_thread(self.state.current_board, self.state.current_thread)
                    return f"Reading Thread {self.state.current_thread} on /biz/:\n\n" + "\n\n".join(
                        f"Post {p['no']}: {self._clean_html(p.get('com', 'No text content'))}"
                        for p in thread_data['posts']
                    )
                except Exception as e:
                    # If there's an error viewing the thread, return to thread list
                    self.state.current_view = "threads"
                    threads_data = self.api.get_catalog(self.state.current_board)
                    return f"Error reading thread, returned to /biz/ Threads\n\n" + "\n\n".join(
                        f"Thread {t['no']}: {self._clean_html(t.get('sub', t.get('com', 'No content'))[:100] + '...')}"
                        for page in threads_data
                        for t in page['threads']
                    )
        except Exception as e:
            return f"Error loading content: {str(e)}"
    
    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the browsing session"""
        player = players[0]  # Single player game
        turn = 0
        
        # Import rich for pretty printing
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.rule import Rule
        
        console = Console()
        
        while turn < self.max_turns:
            console.clear()
            
            # Get current state
            current_state = self.get_current_state()
            state_message = {
                "role": "system",
                "content": current_state
            }
            
            # Display current state in a nice panel
            console.print(Rule("📺 Current View", style="cyan"))
            console.print(Panel(
                current_state,
                title="Model's Current View",
                border_style="cyan",
                padding=(1, 2)
            ))
            
            # Get player's action
            try:
                response = player.get_response(state_message, None)
                outcome = self.attempt_action(response)
                
                # Display model's action in a different panel
                console.print(Rule("🤖 Model's Action", style="magenta"))
                action_text = Text()
                action_text.append(f"Command: ", style="bold cyan")
                action_text.append(response, style="yellow")
                
                if not outcome["valid"]:
                    action_text.append(f"\nError: {outcome['message']}", style="red")
                else:
                    action_text.append(f"\nResult: {outcome['message']}", style="green")
                
                console.print(Panel(
                    action_text,
                    title="Model Response",
                    border_style="magenta",
                    padding=(1, 2)
                ))
                
                if not outcome["valid"]:
                    console.input("\nPress Enter to continue...")
                    continue
                
                if outcome["end_turn"]:
                    turn += 1
                    console.input("\nPress Enter to continue...")
                
                if "game_over" in outcome and outcome["game_over"]:
                    break
            
            except Exception as e:
                console.print(f"[red]Error during turn: {str(e)}[/red]")
                console.input("\nPress Enter to continue...")
                continue
        
        return self.get_result()
    
    def get_result(self) -> Dict:
        """Get browsing session results"""
        return {
            "status": "complete",
            "turns_taken": len(self.state.history),
            "final_view": self.state.current_view,
            "final_board": self.state.current_board,
            "final_thread": self.state.current_thread
        }
