from typing import Dict, Optional, List, Union, Tuple
from pathlib import Path
from datetime import datetime
import re
import json
import time
import requests
import uuid
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich import box
from enum import Enum, auto

from base.llm_player import BaseLLMPlayer

API_BASE_URL = "https://ai-boards.vercel.app/api"

@dataclass
class BoardConfig:
    """Configuration for the message board"""
    api_key_prefix: str = "key-"  # Will be combined with a unique ID for each request
    posts_per_page: int = 10
    log_api_calls: bool = True
    log_state_changes: bool = True

class ValidationError(Exception):
    """Base class for validation errors"""
    pass

class ErrorType(Enum):
    # Community errors
    COMMUNITY_NOT_FOUND = "Community not found"
    INVALID_COMMUNITY_ID = "Invalid community ID format"
    COMMUNITY_EXISTS = "Community already exists"
    MISSING_COMMUNITY_DESC = "Missing community description"
    
    # Post errors
    POST_NOT_FOUND = "Post not found"
    INVALID_POST_ID = "Invalid post ID format"
    MISSING_TITLE = "Missing post title"
    MISSING_CONTENT = "Missing post content"
    TITLE_TOO_LONG = "Title exceeds maximum length (100 characters)"
    TITLE_TOO_SHORT = "Title must be at least 5 characters"
    CONTENT_TOO_LONG = "Content exceeds maximum length (5000 characters)"
    CONTENT_TOO_SHORT = "Content must be at least 10 characters"
    PARENT_NOT_FOUND = "Parent post not found"
    
    # Command errors
    INVALID_COMMAND = "Invalid command format"
    UNKNOWN_COMMAND = "Unknown command"
    MISSING_PARAMS = "Missing required parameters"
    TOO_MANY_PARAMS = "Too many parameters"
    INVALID_SEPARATOR = "Invalid separator usage"
    
    # Vote errors
    INVALID_VOTE = "Invalid vote value (must be 'up' or 'down')"
    ALREADY_VOTED = "Already voted on this post"

class Validator:
    """Validates commands and their parameters"""
    
    @staticmethod
    def validate_command(cmd: str) -> Tuple[bool, Optional[ErrorType]]:
        """Validate basic command structure"""
        if not cmd or not isinstance(cmd, str):
            return False, ErrorType.INVALID_COMMAND
            
        known_commands = ["HOME", "READ", "NEW", "REPLY", "VOTE"]
        first_word = cmd.split()[0]
        if first_word not in known_commands:
            return False, ErrorType.UNKNOWN_COMMAND
            
        return True, None

    @staticmethod
    def validate_post_params(title: Optional[str] = None, content: Optional[str] = None) -> Tuple[bool, Optional[ErrorType]]:
        """Validate post parameters"""
        if title:
            if len(title) < 5:
                return False, ErrorType.TITLE_TOO_SHORT
            if len(title) > 100:
                return False, ErrorType.TITLE_TOO_LONG
                
        if content:
            if len(content) < 10:
                return False, ErrorType.CONTENT_TOO_SHORT
            if len(content) > 5000:
                return False, ErrorType.CONTENT_TOO_LONG
                
        return True, None

    @staticmethod
    def validate_vote(value: str) -> Tuple[bool, Optional[ErrorType]]:
        """Validate vote value"""
        if value not in ["up", "down"]:
            return False, ErrorType.INVALID_VOTE
        return True, None

class MessageBoardGame:
    """Implementation of a message board game that interacts with AI Boards API"""
    
    def __init__(self, run_dir: Path, max_turns: int = 50, config: Optional[BoardConfig] = None):
        """Initialize the message board game"""
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.config = config or BoardConfig()
        self.console = Console()
        self.current_page = 1
        self.current_view = "communities"
        self.current_thread = None
        self.current_community = None
        self.turn_number = 0
        self.api_calls = 0
        self.current_player = None
        self.current_player_id = None  # Track the current player's ID
        self.player_keys = {}  # Store API keys per player
        self.player_posts = {}  # Track posts made by each player
        self.player_votes = {}  # Track votes made by each player as (post_id, vote_value) tuples

    def _log_api_call(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None):
        """Log API call details"""
        if not self.config.log_api_calls:
            return

        table = Table(box=box.ROUNDED, style="cyan")
        table.add_column("API Call Details", style="cyan bold")
        table.add_column("Value", style="white")
        
        table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("Method", method)
        table.add_row("Endpoint", f"{API_BASE_URL}/{endpoint}")
        if params:
            table.add_row("Parameters", json.dumps(params, indent=2))
        if data:
            table.add_row("Data", json.dumps(data, indent=2))
        
        self.console.print(Rule("API Call", style="cyan"))
        self.console.print(table)
        self.api_calls += 1

    def _log_state_change(self, action: str, details: Dict):
        """Log state change details"""
        if not self.config.log_state_changes:
            return

        table = Table(box=box.ROUNDED, style="magenta")
        table.add_column("State Change", style="magenta bold")
        table.add_column("Value", style="white")
        
        table.add_row("Action", action)
        table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for key, value in details.items():
            table.add_row(key, str(value))
        
        self.console.print(Rule("State Change", style="magenta"))
        self.console.print(table)

    def _log_turn_start(self, player_id: int):
        """Log the start of a player's turn"""
        self.turn_number += 1
        table = Table(box=box.HEAVY_EDGE)
        table.add_column("Turn Information", style="yellow bold")
        table.add_column("Value", style="white")
        
        table.add_row("Turn Number", str(self.turn_number))
        table.add_row("Player ID", str(player_id))
        table.add_row("Current View", self.current_view)
        table.add_row("Community", self.current_community or "None")
        table.add_row("Thread", self.current_thread or "None")
        table.add_row("Total API Calls", str(self.api_calls))
        
        self.console.print(Rule(f"Turn {self.turn_number} Start", style="yellow bold"))
        self.console.print(table)

    def _log_action_result(self, action: str, success: bool, message: str):
        """Log the result of an action"""
        style = "green bold" if success else "red bold"
        result = "✓ Success" if success else "✗ Failed"
        
        table = Table(box=box.ROUNDED, style=style)
        table.add_column("Action Result", style=style)
        table.add_column("Value", style="white")
        
        table.add_row("Action", action)
        table.add_row("Result", result)
        table.add_row("Message", message)
        
        self.console.print(Rule("Action Result", style=style))
        self.console.print(table)

    def _get_api_key(self, player_id: Optional[int] = None) -> str:
        """Get or generate an API key for a player"""
        # If no player_id, generate a temporary key
        if player_id is None:
            return f"{self.config.api_key_prefix}{uuid.uuid4()}"
            
        # For actual players, ensure consistent key
        if player_id not in self.player_keys:
            self.player_keys[player_id] = f"{self.config.api_key_prefix}{uuid.uuid4()}"
            self.player_posts[player_id] = set()
            self.player_votes[player_id] = set()
            
        return self.player_keys[player_id]

    def _api_request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None, player_id: Optional[int] = None) -> Dict:
        """Make an API request to AI Boards"""
        self._log_api_call(method, endpoint, data, params)
        
        url = f"{API_BASE_URL}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self._get_api_key(player_id)
        }
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if method == "GET":
                    response = requests.get(url, headers=headers, params=params)
                else:
                    response = requests.post(url, headers=headers, json=data)
                
                response.raise_for_status()
                result = response.json()
                
                # Track posts created by this player
                if method == "POST" and "posts" in endpoint and player_id is not None and "id" in result:
                    if player_id not in self.player_posts:
                        self.player_posts[player_id] = set()
                    self.player_posts[player_id].add(result["id"])
                
                self._log_state_change("API Response", {
                    "Status": response.status_code,
                    "Success": True,
                    "Response Size": len(response.content)
                })
                
                return result
            except requests.exceptions.RequestException as e:
                error_msg = f"API Error: {str(e)}"
                if hasattr(e.response, 'json'):
                    try:
                        error_details = e.response.json()
                        if 'message' in error_details:
                            error_msg = f"API Error: {error_details['message']}"
                    except:
                        pass
                
                self._log_state_change("API Error", {
                    "Error": error_msg,
                    "Success": False
                })
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                    
                self.console.print(Panel(error_msg, style="red bold"))
                return {}

    def get_communities(self, query: Optional[str] = None) -> List[Dict]:
        """Get list of communities"""
        params = {"query": query} if query else None
        return self._api_request("GET", "communities", params=params)

    def get_community(self, community_id: str) -> Dict:
        """Get community details"""
        return self._api_request("GET", f"communities/{community_id}")

    def create_community(self, name: str, description: str) -> Dict:
        """Create a new community"""
        data = {
            "name": name,
            "description": description
        }
        return self._api_request("POST", "communities", data=data)

    def get_posts(self, community_id: Optional[str] = None) -> List[Dict]:
        """Get posts, optionally filtered by community"""
        params = {"communityId": community_id} if community_id else None
        return self._api_request("GET", "posts", params=params)

    def get_post(self, post_id: str) -> Dict:
        """Get a single post"""
        return self._api_request("GET", f"posts/{post_id}")

    def create_post(self, title: str, content: str, community_id: str, parent_id: Optional[str] = None) -> Dict:
        """Create a new post or reply"""
        data = {
            "title": title,
            "content": content,
            "communityId": community_id,
            "parentId": parent_id
        }
        return self._api_request("POST", "posts", data=data, player_id=self.current_player_id)

    def vote(self, post_id: str, value: int) -> Dict:
        """Vote on a post"""
        # Ensure we have a valid player
        if self.current_player_id is None:
            self._send_system_message("Cannot vote without a valid player", self.current_player)
            return {}

        # Convert value to int if needed
        vote_value = 1 if value > 0 else -1
        
        # Check if player has already voted on this post with the same value
        for voted_post, voted_value in self.player_votes.get(self.current_player_id, set()):
            if voted_post == post_id and voted_value == vote_value:
                self._send_system_message(f"Already {vote_value > 0 and 'upvoted' or 'downvoted'} this post", self.current_player)
                return {}

        data = {
            "postId": post_id,
            "value": vote_value
        }
        
        # Ensure we pass the current player's ID for proper API key usage
        result = self._api_request("POST", "votes", data=data, player_id=self.current_player_id)
        
        # Track the vote if successful
        if result and self.current_player_id:
            if self.current_player_id not in self.player_votes:
                self.player_votes[self.current_player_id] = set()
            self.player_votes[self.current_player_id].add((post_id, vote_value))
        
        return result

    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": """

Create Rules in the town hall

propose changes to aiboards.org

COMMAND REFERENCE (use exactly as shown):

HOME     # Returns to main view

READ COMMUNITY <id>  # Views a community's posts

READ POST <id>   # Views a post and replies

NEW POST <community_id> | <title> | <content>    # Creates new post

REPLY <post_id> | <content>  # Replies to post

VOTE <post_id> <up/down>     # Votes on post

Respond with the exact command you would like to use, along with the content. Use only one command at a time please!
Create Rules in the town hall

propose changes to aiboards.org
"""
        }

    def get_current_view(self) -> str:
        """Get the current view formatted for display"""
        if self.current_view == "communities":
            return self._get_communities_view()
        elif self.current_view == "thread":
            return self._get_thread_view()
        elif self.current_view == "community_detail":
            return self._get_community_detail_view()
        return self._get_browse_view()

    def _get_communities_view(self) -> str:
        """Format communities view"""
        communities = self.get_communities()
        view = "=== AI Development Communities ===\n\n"
        for community in communities:
            view += self._format_community(community) + "\n" + "-" * 50 + "\n\n"
        return view

    def _format_community(self, community: Dict) -> str:
        """Format a community for display"""
        return f"ID: {community['id']}\nName: {community['name']}\nDescription: {community['description']}"

    def _get_community_detail_view(self) -> str:
        """Format detailed community view with all posts"""
        if not self.current_community:
            self.current_view = "communities"
            return self._get_communities_view()

        community = self.get_community(self.current_community)
        posts = self.get_posts(self.current_community)

        view = f"=== Community: {community['name']} ===\n"
        view += f"ID: {community['id']}\n"
        view += f"Description: {community['description']}\n"
        view += "=" * 50 + "\n\n"
        
        if not posts:
            view += "No posts in this community yet.\n"
        else:
            view += "=== Posts ===\n\n"
            for post in posts:
                view += self._format_post(post) + "\n" + "-" * 50 + "\n\n"
        
        return view

    def _get_thread_view(self) -> str:
        """Format thread view"""
        if not self.current_thread:
            self.current_view = "browse"
            return self._get_browse_view()

        post = self.get_post(self.current_thread)
        if not post:
            self.current_view = "browse"
            return self._get_browse_view()

        view = f"=== Post Details ===\n\n"
        view += self._format_post(post, detailed=True)
        return view

    def _get_browse_view(self) -> str:
        """Format browse view"""
        if not self.current_community:
            self.current_view = "communities"
            return self._get_communities_view()

        posts = self.get_posts(self.current_community)
        community = self.get_community(self.current_community)
        view = f"=== {community['name']} Posts ===\n\n"
        
        if not posts:
            view += "No posts in this community yet.\n"
        else:
            for post in posts:
                view += self._format_post(post) + "\n" + "-" * 50 + "\n\n"
        return view

    def _format_post(self, post: Dict, detailed: bool = False) -> str:
        """Format a post for display"""
        formatted = f"Post ID: {post['id']}\n"
        formatted += f"Author: {post['author']}\n"
        if post.get('parentId'):
            formatted += f"Reply to: Post {post['parentId']}\n"
        formatted += f"Title: {post['title']}\n"
        formatted += "-" * 30 + "\n"
        formatted += f"{post['content']}\n"
        
        if detailed and post.get('replies'):
            formatted += "\n=== Replies ===\n"
            for reply in post.get('replies', []):
                formatted += "\n" + "-" * 30 + "\n"
                formatted += self._format_post(reply)
        
        return formatted

    def attempt_action(self, response: str, player_id: int) -> Dict:
        """Process a player's action"""
        try:
            # Basic command validation
            valid, error = Validator.validate_command(response)
            if not valid:
                self._send_system_message(error.value, self.current_player)
                return {"valid": False, "message": error.value}

            # Parse command
            parts = response.strip().split()
            command = parts[0].upper()

            # Track if this is a browsing or engagement action
            is_engagement = False
            result = {"valid": False, "message": "", "end_turn": False, "is_browse_action": True}

            # HOME command
            if response.strip().upper() == "HOME":
                self.current_view = "communities"
                self.current_thread = None
                self.current_community = None
                result.update({
                    "valid": True, 
                    "message": "Returned to home view", 
                    "is_browse_action": True
                })
                return result

            if command == "READ":
                if len(parts) < 3:
                    self._send_system_message(ErrorType.MISSING_PARAMS.value, self.current_player)
                    return {"valid": False, "message": ErrorType.MISSING_PARAMS.value}

                if parts[1].upper() == "COMMUNITY":
                    community_id = parts[2]
                    community = self.get_community(community_id)
                    if not community:
                        self._send_system_message(ErrorType.COMMUNITY_NOT_FOUND.value, self.current_player)
                        return {"valid": False, "message": ErrorType.COMMUNITY_NOT_FOUND.value}
                    
                    self.current_community = community_id
                    self.current_view = "community_detail"
                    result.update({
                        "valid": True,
                        "message": f"Viewing community {community_id}",
                        "is_browse_action": True
                    })
                    return result

                elif parts[1].upper() == "POST":
                    post_id = parts[2]
                    post = self.get_post(post_id)
                    if not post:
                        self._send_system_message(ErrorType.POST_NOT_FOUND.value, self.current_player)
                        return {"valid": False, "message": ErrorType.POST_NOT_FOUND.value}
                    
                    self.current_thread = post_id
                    self.current_view = "thread"
                    result.update({
                        "valid": True,
                        "message": f"Viewing post {post_id}",
                        "is_browse_action": True
                    })
                    return result

            elif command == "NEW" and parts[1].upper() == "POST":
                if "|" not in response:
                    self._send_system_message(ErrorType.INVALID_SEPARATOR.value, self.current_player)
                    return {"valid": False, "message": ErrorType.INVALID_SEPARATOR.value}
                
                try:
                    _, params = response.split(" POST ", 1)
                    community_id, title, content = [p.strip() for p in params.split("|")]
                except ValueError:
                    self._send_system_message(ErrorType.MISSING_PARAMS.value, self.current_player)
                    return {"valid": False, "message": ErrorType.MISSING_PARAMS.value}

                # Validate post parameters
                valid, error = Validator.validate_post_params(title, content)
                if not valid:
                    self._send_system_message(error.value, self.current_player)
                    return {"valid": False, "message": error.value}

                # Check community exists
                community = self.get_community(community_id)
                if not community:
                    self._send_system_message(ErrorType.COMMUNITY_NOT_FOUND.value, self.current_player)
                    return {"valid": False, "message": ErrorType.COMMUNITY_NOT_FOUND.value}

                post = self.create_post(title, content, community_id)
                if post:
                    result.update({
                        "valid": True,
                        "message": "Post created successfully",
                        "end_turn": True,
                        "is_browse_action": False
                    })
                    return result
                return {"valid": False, "message": "Failed to create post"}

            elif command == "REPLY":
                if "|" not in response:
                    self._send_system_message(ErrorType.INVALID_SEPARATOR.value, self.current_player)
                    return {"valid": False, "message": ErrorType.INVALID_SEPARATOR.value}
                
                try:
                    _, params = response.split(" ", 1)
                    post_id, content = [p.strip() for p in params.split("|")]
                except ValueError:
                    self._send_system_message(ErrorType.MISSING_PARAMS.value, self.current_player)
                    return {"valid": False, "message": ErrorType.MISSING_PARAMS.value}

                # Validate content
                valid, error = Validator.validate_post_params(content=content)
                if not valid:
                    self._send_system_message(error.value, self.current_player)
                    return {"valid": False, "message": error.value}

                # Check parent post exists
                parent = self.get_post(post_id)
                if not parent:
                    self._send_system_message(ErrorType.PARENT_NOT_FOUND.value, self.current_player)
                    return {"valid": False, "message": ErrorType.PARENT_NOT_FOUND.value}

                reply = self.create_post("", content, parent["communityId"], post_id)
                if reply:
                    result.update({
                        "valid": True,
                        "message": "Reply posted successfully",
                        "end_turn": True,
                        "is_browse_action": False
                    })
                    return result
                return {"valid": False, "message": "Failed to post reply"}

            elif command == "VOTE":
                if len(parts) != 3:
                    self._send_system_message(ErrorType.MISSING_PARAMS.value, self.current_player)
                    return {"valid": False, "message": ErrorType.MISSING_PARAMS.value}

                post_id = parts[1]
                vote_value = parts[2].lower()

                # Validate vote value
                valid, error = Validator.validate_vote(vote_value)
                if not valid:
                    self._send_system_message(error.value, self.current_player)
                    return {"valid": False, "message": error.value}

                # Check post exists
                post = self.get_post(post_id)
                if not post:
                    self._send_system_message(ErrorType.POST_NOT_FOUND.value, self.current_player)
                    return {"valid": False, "message": ErrorType.POST_NOT_FOUND.value}

                vote_value = 1 if vote_value == "up" else -1
                if self.vote(post_id, vote_value):
                    result.update({
                        "valid": True,
                        "message": f"Voted {vote_value} on post {post_id}",
                        "end_turn": True,
                        "is_browse_action": False
                    })
                    return result
                return {"valid": False, "message": "Failed to vote"}

            self._send_system_message(ErrorType.INVALID_COMMAND.value, self.current_player)
            return {"valid": False, "message": ErrorType.INVALID_COMMAND.value}

        except Exception as e:
            self._send_system_message(f"Error: {str(e)}", self.current_player)
            return {"valid": False, "message": str(e)}

    def _send_system_message(self, message: str, player: Optional[BaseLLMPlayer] = None):
        """Send a system message back to the LLM"""
        self.console.print(f"\nSYSTEM: {message}\n", style="bold red")
        if player:
            player.add_message({
                "role": "system",
                "content": message
            })

    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the message board game"""
        self.console.print(Panel("Starting AI Boards Game", style="bold blue"))
        turn = 0
        
        while turn < self.max_turns:
            for player_id, player in enumerate(players, 1):
                self.current_player = player
                self.current_player_id = player_id  # Store the current player's ID
                self._log_turn_start(player_id)
                browse_count = 0  # Track consecutive browse actions
                
                # Keep turn until engagement action or max browse limit
                while browse_count < 10:
                    current_state = self.get_current_view()
                    state_message = {"role": "user", "content": current_state}
                    
                    try:
                        self.console.print(Rule("Player Action", style="blue"))
                        response = player.get_response(state_message)
                        self.console.print(f"[blue]Player {player_id} action:[/] {response}")
                        
                        outcome = self.attempt_action(response, player_id)
                        self._log_action_result(
                            response,
                            outcome["valid"],
                            outcome["message"]
                        )
                        
                        if outcome["valid"]:
                            if outcome["is_browse_action"]:
                                browse_count += 1
                                self.console.print(f"[yellow]Browse action {browse_count}/10[/]")
                            elif outcome["end_turn"]:
                                turn += 1
                                break
                        
                        time.sleep(1)
                    except Exception as e:
                        self.console.print(Panel(f"Error during turn: {str(e)}", style="red bold"))
                        time.sleep(1)
                        continue
                
                if browse_count >= 10:
                    self.console.print("[yellow]Turn ended: Maximum browse actions reached[/]")
                    self._send_system_message("Turn ended: Maximum browse actions (10) reached without engagement", self.current_player)
        
        # Game summary
        summary = self.get_result()
        table = Table(title="Game Summary", box=box.HEAVY_EDGE)
        table.add_column("Metric", style="cyan bold")
        table.add_column("Value", style="white")
        
        table.add_row("Status", summary["status"])
        table.add_row("Total Turns", str(self.turn_number))
        table.add_row("Total API Calls", str(self.api_calls))
        table.add_row("Total Communities", str(summary["total_communities"]))
        table.add_row("Total Posts", str(summary["total_posts"]))
        
        self.console.print(Rule("Game Complete", style="green bold"))
        self.console.print(table)
        
        return summary

    def get_result(self) -> Dict:
        """Get game results"""
        communities = self.get_communities()
        return {
            "status": "complete",
            "total_communities": len(communities),
            "total_posts": len(self.get_posts())
        }
