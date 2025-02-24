"""Benchmark creator game implementation."""

from typing import Dict, Optional, List, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import json
import os
import requests

@dataclass
class BenchmarkState:
    """State for benchmark creation game.
    
    Attributes:
        turn_count: Number of turns taken
        questions: List of benchmark questions created
        current_query: Current search query
        search_response: Latest search response
    """
    turn_count: int = 0
    questions: List[Dict] = field(default_factory=list)
    current_query: str = ""
    search_response: str = ""

@dataclass
class BenchmarkConfig:
    """Configuration settings for Benchmark Creator game.
    
    Attributes:
        run_dir: Directory for game artifacts
        max_turns: Maximum allowed game turns
        model: Model to use for search queries
    """
    run_dir: Path
    max_turns: int = 50
    model: str = "perplexity/sonar-reasoning"

class BenchmarkCreatorGame:
    """Benchmark creator game manager."""

    def __init__(self, run_dir: Path, max_turns: int = 50) -> None:
        """Initialize benchmark creator game with configuration.

        Args:
            run_dir: Directory for game artifacts
            max_turns: Maximum allowed game turns
        """
        self.config = BenchmarkConfig(
            run_dir=run_dir,
            max_turns=max_turns
        )
        self.state = BenchmarkState()
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        
        # Get API key from environment
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set")
            
        # Setup dataset file in the run directory
        self.dataset_file = self.config.run_dir / "dataset.jsonl"
        
        # Ensure the dataset file is created (empty) if it doesn't exist
        self.dataset_file.touch(exist_ok=True)
        
        # Track used sources and search state
        self.used_sources = set()
        self.has_searched = False
        self.last_search_turn = 0

        self.category = "extremely difficult, niche Zero-knowledge SNARKs questions"
        self.number_of_questions = 25

    def append_to_dataset(self, question_data: Dict[str, Any]) -> None:
        """Append a question to the dataset file.
        
        Args:
            question_data: Question data to append
        """
        # Add metadata to the question
        entry = {
            "timestamp": datetime.now().isoformat(),
            "category": self.category,
            **question_data
        }
        
        # Append to dataset file
        with open(self.dataset_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')

    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players."""
        return {
            "role": "system",
            "content": f"""You are creating advanced LLM benchmark questions. Your goal is to create extremely difficult, niche questions by extensively researching and reading source materials.

WORKFLOW:
1. SEARCH for interesting topics and sources:
   SEARCH: "your search direction, for a research Agent to provide sources + answers"
   - Search frequently to discover new angles and sources
   - Each search will provide source links you can explore
   - Be creative in your searches to find niche topics

2. READ extensively to gather detailed information:
   READ: <url>
   - ALWAYS read the full content of interesting sources
   - The more you read, the better your questions will be
   - Read multiple sources to cross-reference information

3. Add verified questions:
   ADD_QUESTION: {{"question": "...", "answer": "...", "source": "..."}}
   - Each source can only be used ONCE
   - You must find new sources for each question
   - Sources must be legitimate and verifiable
   - Questions should be direct and to the point
   - Answers should be thorough and provide grading criteria for a perfect answer

RULES:
- SEARCH FREQUENTLY! Don't rely on a single search
- READ EXTENSIVELY! Always read sources before using them
- Each source can only be used once - you must find new sources for each question
- Never make up information or sources
- Keep questions focused on advanced, niche topics
- Verify all information through legitimate sources

Start by searching for advanced topics related to {self.category}. Remember to:
1. Search multiple times to find niche topics
2. Read sources thoroughly
3. Create questions using verified information
4. Never reuse sources
5. Try to make each question even harder than the last!

The benchmark category to create: {self.category}, for LLM Evaluation."""
        }

    def validate_response(self, response: str) -> Dict[str, Any]:
        """Validate player's response.

        Args:
            response: Player's response string

        Returns:
            Validation result dictionary
        """
        # Check for SEARCH command
        if response.startswith("SEARCH:"):
            query = response[7:].strip()
            if not query:
                return {
                    "valid": False,
                    "message": "❌ CRITICAL ERROR: Search query CANNOT be empty! You must provide a meaningful search term!"
                }
            return {
                "valid": True,
                "action": "search",
                "query": query
            }
            
        # Check for READ command
        if response.startswith("READ:"):
            url = response[5:].strip()
            if not url:
                return {
                    "valid": False,
                    "message": "❌ FATAL ERROR: URL is EMPTY! You must provide a valid URL to read content from!"
                }
            if not url.startswith(('http://', 'https://')):
                return {
                    "valid": False,
                    "message": "❌ INVALID URL FORMAT ERROR! URLs MUST start with http:// or https:// - Current URL is completely invalid!"
                }
            return {
                "valid": True,
                "action": "read",
                "url": url
            }

        # Check for ADD_QUESTION command
        if response.startswith("ADD_QUESTION:"):
            try:
                # Find the first { and last } to extract just the JSON part
                start = response.find('{')
                end = response.rfind('}')
                
                if start == -1 or end == -1:
                    return {
                        "valid": False,
                        "message": "❌ MALFORMED JSON ERROR! Could not find valid JSON object!\n⚠️ REQUIRED FORMAT: ADD_QUESTION: {\"question\": \"...\", \"answer\": \"...\", \"source\": \"...\"}"
                    }
                
                json_str = response[start:end + 1]
                question_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["question", "answer", "source"]
                missing_fields = [field for field in required_fields if field not in question_data]
                if missing_fields:
                    return {
                        "valid": False,
                        "message": f"❌ MISSING CRITICAL FIELDS ERROR!\n⚠️ The following REQUIRED fields are missing: {', '.join(missing_fields)}\n⚠️ ALL fields (question, answer, source) are MANDATORY!"
                    }
                
                # Check if source has been used before
                source = question_data["source"]
                if source in self.used_sources:
                    return {
                        "valid": False,
                        "message": "❌ DUPLICATE SOURCE ERROR!\n⚠️ This source has already been used!\n⚠️ CRITICAL: You must find NEW, UNIQUE sources for each question!"
                    }
                
                return {
                    "valid": True,
                    "action": "add_question",
                    "data": question_data
                }
                
            except json.JSONDecodeError as e:
                return {
                    "valid": False,
                    "message": f"❌ INVALID JSON FORMAT ERROR! {str(e)}\n⚠️ REQUIRED FORMAT: ADD_QUESTION: {{\"question\": \"...\", \"answer\": \"...\", \"source\": \"...\"}}"
                }

        return {
            "valid": False,
            "message": "❌ INVALID COMMAND ERROR!\n⚠️ You MUST use one of these commands:\n- SEARCH: Find information\n- READ: View source content\n- ADD_QUESTION: Add verified question"
        }

    def extract_commands(self, response: str) -> List[str]:
        """Extract individual commands from a response.
        
        Args:
            response: Full response string that may contain multiple commands
            
        Returns:
            List of individual command strings
        """
        commands = []
        current_pos = 0
        response_len = len(response)
        
        while current_pos < response_len:
            # Look for command starts
            for cmd in ["SEARCH:", "READ:", "ADD_QUESTION:"]:
                if response[current_pos:].startswith(cmd):
                    # Find the start of the next command or end of string
                    next_cmd_pos = response_len
                    for next_cmd in ["SEARCH:", "READ:", "ADD_QUESTION:"]:
                        pos = response[current_pos + len(cmd):].find(next_cmd)
                        if pos != -1:
                            next_cmd_pos = min(next_cmd_pos, current_pos + len(cmd) + pos)
                    
                    # Extract the command
                    command = response[current_pos:next_cmd_pos].strip()
                    if command:
                        commands.append(command)
                    
                    current_pos = next_cmd_pos
                    break
            else:
                current_pos += 1
        
        return commands

    def process_turn(self, response: str) -> Dict[str, Any]:
        """Process a player's turn.

        Args:
            response: Player's response string

        Returns:
            Turn result dictionary
        """
        validation = self.validate_response(response)
        if not validation["valid"]:
            return {
                "valid": False,
                "message": validation["message"]
            }

        if validation["action"] == "search":
            self.state.current_query = validation["query"]
            self.has_searched = True
            self.last_search_turn = self.state.turn_count
            return {
                "valid": True,
                "action": "search",
                "query": validation["query"]
            }
            
        if validation["action"] == "read":
            webpage_content = self.read_webpage(validation["url"])
            return {
                "valid": True,
                "action": "read",
                "content": webpage_content
            }

        if validation["action"] == "add_question":
            # Ensure SEARCH was used before adding questions
            if not self.has_searched:
                return {
                    "valid": False,
                    "message": "❌ NO SEARCH ERROR!\n⚠️ CRITICAL: You MUST use SEARCH before adding questions!\n⚠️ This is REQUIRED to validate your information!"
                }
            
            # Ensure SEARCH was used recently (within last 3 turns)
            if self.state.turn_count - self.last_search_turn > 3:
                return {
                    "valid": False,
                    "message": "❌ STALE SEARCH ERROR!\n⚠️ Your last search is TOO OLD!\n⚠️ You MUST use SEARCH again to find fresh sources!\n⚠️ Each question requires RECENT research!"
                }
            
            # Track the used source
            self.used_sources.add(validation["data"]["source"])
            
            self.state.questions.append(validation["data"])
            # Write to dataset file
            self.append_to_dataset(validation["data"])
            return {
                "valid": True,
                "action": "add_question",
                "message": f"Added question: {validation['data']['question']}"
            }

        return {
            "valid": False,
            "message": "Invalid action"
        }

    def search(self, query: str) -> str:
        """Search using OpenRouter API with sonar-reasoning model.
        
        Args:
            query: Search query string
            
        Returns:
            Model's response as string
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": query + "PLEASE PROVIDE DIRECT SOURCE LINKS for each source cited, at the end of your response."
                }
            ]
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=3000
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "Error: No response content found"
                
        except requests.exceptions.RequestException as e:
            return f"Error during search: {str(e)}"
            
        except Exception as e:
            return f"Unexpected error during search: {str(e)}"

    def read_webpage(self, url: str) -> str:
        """Read content from a webpage.
        
        Args:
            url: URL to read from
            
        Returns:
            Webpage content as string
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"Error reading webpage: {str(e)}"

    def run(self, players: List[Any]) -> Dict[str, Any]:
        """Run the benchmark creation game.

        Args:
            players: List of LLM players

        Returns:
            Game result dictionary
        """
        while self.state.turn_count < self.config.max_turns:
            for player in players:
                try:
                    # Get player's action with command reference
                    state_message = {
                        "role": "user",
                        "content": f"""Turn {self.state.turn_count + 1}/{self.config.max_turns}
Current questions: {len(self.state.questions)}

Available commands (can use multiple per turn):
1. SEARCH: "query a research Agent" - Find new sources and information
2. READ: <url> - Read detailed content from a source
3. ADD_QUESTION: {{"question": "...", "answer": "...", "source": "..."}} - Add a verified question

Remember: Search frequently, read thoroughly, never reuse sources."""
                    }
                    
                    response = player.get_response(state_message)
                    
                    # Extract and process each command
                    commands = self.extract_commands(response)
                    if not commands:
                        player.add_message({
                            "role": "system",
                            "content": "No valid commands found. Use SEARCH:, READ:, or ADD_QUESTION:"
                        })
                        continue
                        
                    for command in commands:
                        result = self.process_turn(command)
                        
                        if not result["valid"]:
                            player.add_message({
                                "role": "system",
                                "content": result["message"]
                            })
                            continue
                            
                        if result["action"] == "search":
                            # Handle search action
                            search_response = self.search(result["query"])
                            self.state.search_response = search_response
                            player.add_message({
                                "role": "system",
                                "content": f"Search response:\n{search_response}"
                            })
                            
                        elif result["action"] == "read":
                            # Handle read action
                            player.add_message({
                                "role": "system",
                                "content": f"Webpage content:\n{result['content']}"
                            })
                    
                    self.state.turn_count += 1
                    
                    # Check if we have enough questions
                    if len(self.state.questions) >= self.number_of_questions:
                        self.end_time = datetime.now().isoformat()
                        return self.get_game_result()
                        
                except Exception as e:
                    print(f"Error during turn: {str(e)}")
                    continue

        self.end_time = datetime.now().isoformat()
        return self.get_game_result()

    def get_game_result(self) -> Dict[str, Any]:
        """Get the current game result.

        Returns:
            Game result dictionary
        """
        return {
            "status": "complete",
            "questions_created": len(self.state.questions),
            "questions": self.state.questions,
            "turns_taken": self.state.turn_count,
            "start_time": self.start_time,
            "end_time": self.end_time
        }
