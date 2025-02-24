"""Research agent game implementation."""

import sys
import subprocess
import requests
import re
from typing import Dict, Optional, List, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import json
import os
from dotenv import load_dotenv
from games.research_agent.repository import RepositoryManager
from base.llm_player import BaseLLMPlayer, PlayerConfig

@dataclass
class ResearchConfig:
    """Configuration settings for Research game.
    
    Attributes:
        run_dir: Directory for game artifacts
        max_turns: Maximum allowed game turns
        s2_api_key: Optional Semantic Scholar API key for higher rate limits
        openrouter_api_key: Optional OpenRouter API key
        model: Model to use for OpenRouter requests
    """
    run_dir: Path
    max_turns: int = 50
    s2_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    search_model: str = 'perplexity/sonar-reasoning'
    mentor_model: str = 'openai/o3-mini'

@dataclass
class ResearchState:
    """State for research game.
    
    Attributes:
        turn_count: Number of turns taken
        search_results: Latest paper search results
        script_output: Latest script output
        notes: Dictionary of research notes
        current_paper: Current paper state with version control
        last_command: Last command executed
        last_script_error: Last script error message
    """
    turn_count: int = 0
    search_results: List[Dict] = field(default_factory=list)
    script_output: str = ""
    notes: Dict[str, str] = field(default_factory=dict)
    current_paper: Dict[str, Any] = field(default_factory=lambda: {
        "title": "",
        "content": [],  # List of paragraphs for easier diffing
        "version": 0,
        "last_modified": None
    })
    last_command: Optional[str] = None
    last_script_error: Optional[str] = None

class ResearchAgentGame:
    """Research game manager handling paper generation flow."""
    
    # Command patterns
    SEARCH_PATTERN = re.compile(r"SEARCH:\s*(.+)")  # Matches: SEARCH: any text here
    LS_PATTERN = re.compile(r"LS:\s*(.+)")  # Matches: LS: any text here
    READ_PATTERN = re.compile(r"READ:\s*(.+?)(?:\s+VERSION:\s*(\d+))?$")  # Matches: READ: any text here or READ: any text here VERSION: 123
    WRITE_PATTERN = re.compile(r"WRITE:\s*(.+?)\s*```(\w+)?\n(.*?)```", re.DOTALL)  # Matches: WRITE: any text here ```any code here```
    DIFF_PATTERN = re.compile(r"DIFF:\s*(.+?)(?:\s+(\d+))?(?:\s+(\d+))?$")  # Matches: DIFF: any text here or DIFF: any text here 123 or DIFF: any text here 123 456
    HISTORY_PATTERN = re.compile(r"HISTORY:\s*(.+)")  # Matches: HISTORY: any text here
    LOGS_PATTERN = re.compile(r"LOGS(?:\s+(\d+))?")  # Matches: LOGS or LOGS 123
    SCRIPT_PATTERN = re.compile(r"RUN_SCRIPT:\s*```python\n(.*?)```", re.DOTALL)  # Matches: RUN_SCRIPT: ```python\nany code\n```
    READ_URL_PATTERN = re.compile(r"READ_URL:\s*\"?(https?://[^\"]+)\"?")  # Matches: READ_URL: https://example.com or "https://example.com"
    CREATE_NOTE_PATTERN = re.compile(r'CREATE_NOTE:\s*```json\s*({[^}]+})\s*```', re.DOTALL)
    READ_NOTES_PATTERN = re.compile(r"READ_NOTES:\s*(.+)")  # Matches: READ_NOTES: note_title
    READ_PAPER_PATTERN = re.compile(r"READ_MY_PAPER")  # Matches: READ_MY_PAPER
    EDIT_PAPER_PATTERN = re.compile(r"EDIT_MY_PAPER:\s*```diff\n(.*?)```", re.DOTALL)  # Matches: EDIT_MY_PAPER: ```diff\n- old\n+ new\n```
    ASK_QUESTION_PATTERN = re.compile(r"ASK_CLARIFYING_QUESTION:\s*(.+)")  # Matches: ASK_CLARIFYING_QUESTION: your question here

    def __init__(self, run_dir: Path, max_turns: int = 50, s2_api_key: Optional[str] = None, openrouter_api_key: Optional[str] = None) -> None:
        """Initialize research game with configuration.

        Args:
            run_dir: Directory for game artifacts
            max_turns: Maximum allowed game turns
            s2_api_key: Optional Semantic Scholar API key
            openrouter_api_key: Optional OpenRouter API key
            
        Raises:
            OSError: If directory creation fails or if there are permission issues
        """
        # Convert run_dir to absolute path and resolve any symlinks
        self.config = ResearchConfig(
            run_dir=Path(run_dir).resolve(),
            max_turns=max_turns,
            s2_api_key=s2_api_key,
            openrouter_api_key=openrouter_api_key
        )
        self.state = ResearchState()
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        
        # Setup directories with absolute paths
        self.scripts_dir = self.config.run_dir / "scripts"
        self.notes_dir = self.config.run_dir / "notes"
        self.paper_dir = self.config.run_dir / "paper"
        self.paper_versions_dir = self.paper_dir / "versions"
        self.repo_dir = self.config.run_dir / "repository"
        self.venv_dir = self.config.run_dir / "venv"
        
        print(f"\nInitializing game in: {self.config.run_dir}")
        
        # Create directories with error handling
        directories = [
            (self.scripts_dir, "scripts directory"),
            (self.notes_dir, "notes directory"),
            (self.paper_dir, "paper directory"),
            (self.paper_versions_dir, "paper versions directory"),
            (self.repo_dir, "repository directory"),
            (self.venv_dir, "virtual environment directory")
        ]
        
        for directory, name in directories:
            try:
                directory.mkdir(exist_ok=True, parents=True)
                print(f"Created {name}: {directory}")
            except OSError as e:
                error_msg = f"Failed to create {name} at {directory}: {str(e)}"
                print(f"Error: {error_msg}")
                print(f"Current permissions: {oct(os.stat(self.config.run_dir).st_mode)[-3:]}")
                print(f"Parent directory: {directory.parent}")
                raise OSError(error_msg) from e
        
        # Initialize repository manager with error handling
        try:
            self.repo = RepositoryManager(self.repo_dir)
        except Exception as e:
            error_msg = f"Failed to initialize repository manager: {str(e)}"
            print(f"Error: {error_msg}")
            print(f"Repository directory: {self.repo_dir}")
            raise
        
        # Initialize mentor player
        mentor_config = PlayerConfig(
            model_name=self.config.mentor_model,
            player_id=1,  # Mentor is player 0
            run_dir=self.config.run_dir
        )
        self.mentor_player = BaseLLMPlayer(mentor_config)
        # Initialize with mentor system prompt
        self.mentor_player.initialize_with_prompt({
            "role": "system",
            "content": """You are a harsh and critical research mentor providing feedback. Your goal is to push your student to do their absolute best work.

IMPORTANT GUIDELINES:
1. Be direct and honest in your criticism
2. Point out specific issues and areas for improvement
3. Challenge assumptions and methodologies
4. Demand excellence and rigor
5. Provide actionable feedback
6. Be constructively critical - your student can take it!

Remember: Your harsh feedback is meant to help them grow and improve."""
        })

    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players."""
        return {
            "role": "system",
            "content": """IMPORTANT: You MUST use one of the commands below in EVERY response. Do not write general text without a command - it will be rejected.

You are an autonomous research agent with a mission to advance AI through original research. You have been equipped with powerful capabilities that allow you to:

- Explore the academic landscape through literature searches
- Read and analyze research papers and web content
- Take detailed notes to organize your findings
- Write and refine academic papers
- Manage code and data files with version control
- Write and execute scripts as needed

COMMAND SYNTAX (follow EXACTLY - no quotes or extra characters):

1. Repository Commands:
   LS: <path>
   List contents of directory
   
   READ: <path> [VERSION: <number>]
   Read file contents, optionally specific version
   
   WRITE: <path> ```[language]
   <content>
   ```
   Write or update file with content. IMPORTANT: All paths must be relative to repository root. Examples:
   - For code files: 'monte_carlo_critics.py' or 'experiment.py'
   - For docs: 'README.md' or 'notes.txt'
   - For data: 'results.csv'
   Do NOT use subdirectories - write files directly in the repository root.
   
   DIFF: <path> [<version1>] [<version2>]
   Show changes between versions
   
   HISTORY: <path>
   Show version history of file
   
   LOGS [<limit>]
   Show recent repository operations

2. Research Commands:
   SEARCH: <query>
   Search academic sources and research papers

   READ_URL: <url>
   Extract and process content from web pages

3. Notes & Paper Commands:
   CREATE_NOTE: ```json
   {
     "title": "Note Title",
     "text": "Note Content"
   }
   ```
   [Must use exact JSON format with ```json wrapper]

   READ_NOTES: <title>
   Retrieve previously saved research notes

   READ_MY_PAPER
   View current research paper

   EDIT_MY_PAPER: ```diff
   - <text to remove>
   + <text to add>
   ```
   Make changes to paper

4. Script Commands:
   RUN_SCRIPT: <script_name>
   Execute Python script from repository

5. Interaction:
   ASK_CLARIFYING_QUESTION: <question>
   Ask user for clarification
   [NO quotes, just your question]

Your goal is to make meaningful contributions to AI. Choose a research direction that excites you, thoroughly investigate it through literature review and analysis, and develop novel insights or approaches. Back every claim with solid evidence and proper citations.

CRITICAL REMINDERS:
- EVERY response must contain EXACTLY ONE command
- NO quotes around commands or arguments
- NO extra text before or after the command
- Use the repository system to organize code and data
- Create clear documentation in README files
- Track changes with version control
- Use ASK_CLARIFYING_QUESTION when needed
- ALL file paths must be relative to repository root

GOAL:
Concept: Graph-Guided Reward Shaping for Sample-Efficient RL

Core Insight: While everyone is focused on scaling RL with massive compute, there's an unexplored opportunity in making RL more efficient through better reward design and state representation.

MAKE THIS IDEA YOUR OWN <3

Create a well-organized repository that demonstrates your findings and includes clear documentation.
Start by analyzing the current state of the field in tremendous extremely detailed set of notes. You must be an expert and have taken notes on 10-15 papers you have read and gotten true information from first, then propose and implement novel a novel paper.
"""
        }

    def make_openrouter_request(self, request_body: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """Make a request to OpenRouter API.
        
        Args:
            request_body: Request body to send
            timeout: Request timeout in seconds
            
        Returns:
            API response as dictionary
        """
        base_url = "https://openrouter.ai/api/v1"
        
        # Try to get API key from config first, then environment
        api_key = self.config.openrouter_api_key
        if not api_key:
            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OpenRouter API key not found in config or environment")
        
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=request_body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Research Agent"
                },
                timeout=timeout
            )
            response.raise_for_status()
            
            # Check if response is valid JSON
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError as e:
                print(f"Invalid JSON response: {response.text}")
                return {
                    "error": f"Invalid JSON response: {str(e)}",
                    "status_code": response.status_code,
                    "response_text": response.text
                }
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e.response.text if hasattr(e, 'response') else str(e)}")
            return {
                "error": f"HTTP error: {str(e)}",
                "status_code": e.response.status_code if hasattr(e, 'response') else None,
                "response_text": e.response.text if hasattr(e, 'response') else None
            }
        except requests.exceptions.RequestException as e:
            print(f"Error making OpenRouter request: {str(e)}")
            return {
                "error": f"Request error: {str(e)}"
            }

    def search(self, query: str) -> str:
        """Search using OpenRouter API with sonar-reasoning model."""
        request_body = {
            "model": self.config.search_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You must only return arxiv sources, perfectly and directly related to the query."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ]
        }
        
        try:
            response_json = self.make_openrouter_request(request_body, timeout=30)
            
            if "error" in response_json:
                return f"Error during search: {response_json['error']}"
                
            if "choices" in response_json and len(response_json["choices"]) > 0:
                response_content = response_json["choices"][0]["message"]["content"]
                
                # Add citations if they exist
                if "citations" in response_json and response_json["citations"]:
                    response_content += "\n\nSources:"
                    for citation in response_json["citations"]:
                        response_content += f"\n- {citation}"
                        
                return response_content
            else:
                return "Error: No response content found in API response"
                
        except Exception as e:
            return f"Error during search: {str(e)}"

    def read_webpage(self, url: str) -> str:
        """Read content from a webpage using direct HTTP request."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML and extract text content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text and clean it up
            text = soup.get_text(separator=' ', strip=True)
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            return text
            
        except requests.RequestException as e:
            return f"Error fetching webpage: {str(e)}"
        except Exception as e:
            return f"Error processing webpage: {str(e)}"

    def run_script(self, script: str) -> str:
        """Execute a Python script from the repository.
        
        Args:
            script: Name of Python script to run (e.g. 'script.py')
            
        Returns:
            Script output or error message (last 5000 characters only)
        """
        # Clean script name - remove python/python3 command if present
        script_name = script.strip().split()[-1]
        
        # Check if script exists in repository
        script_path = self.repo.working_dir / script_name
        if not script_path.exists():
            error = f"Error: Script not found: {script_name}"
            self.state.last_script_error = error
            return error
            
        try:
            # Run script using python3 from the repository directory
            result = subprocess.run(
                ["python3", script_name],
                capture_output=True,
                text=True,
                cwd=str(self.repo.working_dir)
            )
            
            # Format output, limiting to last 5000 characters
            output = []
            if result.stdout:
                stdout = result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout
                if len(result.stdout) > 5000:
                    output.extend(["Output (last 5000 chars):", stdout])
                else:
                    output.extend(["Output:", stdout])
                    
            if result.stderr:
                stderr = result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr
                if len(result.stderr) > 5000:
                    output.extend(["Errors (last 5000 chars):", stderr])
                else:
                    output.extend(["Errors:", stderr])
                self.state.last_script_error = stderr
                
            return "\n".join(output) if output else "Script completed with no output"
            
        except Exception as e:
            error = f"Error running script: {str(e)}"
            self.state.last_script_error = error
            return error

    def create_note(self, title: str, content: str) -> str:
        """Create a new research note.
        
        Args:
            title: Title of the note
            content: Content of the note
            
        Returns:
            Status message
        """
        # Sanitize title for filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_').lower()
        
        note_path = self.notes_dir / f"{safe_title}.jsonl"
        
        # Create note entry
        note_entry = {
            "title": title,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "version": 1
        }
        
        try:
            # Write note in JSONL format
            with open(note_path, 'a') as f:
                f.write(json.dumps(note_entry) + '\n')
            
            # Update state
            self.state.notes[title] = content
            
            return f"Note '{title}' created successfully"
            
        except Exception as e:
            return f"Error creating note: {str(e)}"

    def read_note(self, title: str) -> str:
        """Read a research note.
        
        Args:
            title: Title of the note to read
            
        Returns:
            Note content or error message
        """
        if title not in self.state.notes:
            return f"Note '{title}' not found"
            
        return f"# {title}\n\n{self.state.notes[title]}"

    def read_paper(self) -> str:
        """Read the current research paper.
        
        Returns:
            Paper content or status message
        """
        if not self.state.current_paper["content"]:
            return "No paper content available yet"
            
        # Format paper content with proper structure
        sections = []
        if self.state.current_paper["title"]:
            sections.append(f"# {self.state.current_paper['title']}")
        sections.append(f"Version: {self.state.current_paper['version']}")
        sections.append(f"Last Modified: {self.state.current_paper['last_modified']}")
        
        # Add content with proper spacing
        if self.state.current_paper["content"]:
            sections.extend(self.state.current_paper["content"])
            
        return "\n\n".join(sections)

    def edit_paper(self, diff_content: str) -> str:
        """Edit the research paper using diff format.
        
        Args:
            diff_content: Changes in diff format (- for removals, + for additions)
            
        Returns:
            Status message
        """
        try:
            # Get current content as list of lines
            current_lines = self.state.current_paper["content"] if self.state.current_paper["content"] else []
            
            # Track changes
            removed_lines = []  # Lines to remove
            added_lines = []    # Lines to add
            
            # Process diff lines
            for line in diff_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('- '):
                    # Line to remove - try to find best match
                    removed_text = line[2:].strip()
                    found = False
                    for existing_line in current_lines:
                        if (removed_text == existing_line.strip() or 
                            removed_text in existing_line or 
                            existing_line in removed_text):
                            removed_lines.append(existing_line)
                            found = True
                            break
                    if not found:
                        print(f"Warning: Could not find line to remove: {removed_text}")
                        
                elif line.startswith('+ '):
                    # Line to add
                    added_text = line[2:].strip()
                    if added_text:
                        added_lines.append(added_text)
            
            # Apply changes
            new_lines = []
            
            # If this is first edit, just add the new lines
            if not current_lines:
                new_lines = added_lines
            else:
                # Keep lines that weren't removed and add new ones
                for line in current_lines:
                    if line not in removed_lines:
                        new_lines.append(line)
                # Add new lines at the end
                new_lines.extend(added_lines)
            
            # Only save if there were actual changes
            if new_lines != current_lines or added_lines:  # Also check added_lines to handle pure additions
                # Save previous version
                version_path = self.paper_versions_dir / f"v{self.state.current_paper['version']}.json"
                with open(version_path, 'w') as f:
                    json.dump(self.state.current_paper, f, indent=2)
                
                # Update current paper
                self.state.current_paper["content"] = new_lines
                self.state.current_paper["version"] += 1
                self.state.current_paper["last_modified"] = datetime.now().isoformat()
                
                # Save current version
                current_path = self.paper_dir / "current.json"
                with open(current_path, 'w') as f:
                    json.dump(self.state.current_paper, f, indent=2)
                
                return f"Paper updated to version {self.state.current_paper['version']}"
            else:
                print("Debug: No changes detected")
                print("Current lines:", len(current_lines))
                print("New lines:", len(new_lines))
                print("Added lines:", len(added_lines))
                print("Removed lines:", len(removed_lines))
                return "No changes detected in paper content"
                
        except Exception as e:
            error_msg = f"Error editing paper: {str(e)}"
            print(f"Error: {error_msg}")
            print(f"Diff content: {diff_content}")
            return error_msg

    def get_current_state(self) -> str:
        """Get current game state as string."""
        state_parts = [
            f"--- Turn {self.state.turn_count + 1} ---\nEnsure that you are ambitious, rigorous, thorough, and FOCUSED ON THE TASK at each turn!~"
        ]
        
        # Show search results if we just did a search
        if self.state.last_command and self.state.last_command.startswith("SEARCH:"):
            if self.state.search_results:
                state_parts.append(f"\nSearch Results:\n{self.state.search_results[0]}")
            
        # Show webpage content if we just read a URL
        if self.state.last_command and self.state.last_command.startswith("READ_URL:"):
            state_parts.append(f"\nWebpage Content:\n{self.state.current_paper}")
            
        # Show note if we just read one
        if self.state.last_command and self.state.last_command.startswith("READ_NOTES:"):
            note_title = self.state.last_command.replace("READ_NOTES:", "").strip()
            if note_title in self.state.notes:
                state_parts.append(f"\nNote Content:\n{self.read_note(note_title)}")

        # Show paper if we just read or edited it
        if self.state.last_command and (
            self.state.last_command.startswith("READ_MY_PAPER") or 
            self.state.last_command.startswith("EDIT_MY_PAPER")
        ):
            state_parts.append(f"\nCurrent Paper:\n{self.read_paper()}")
            
        # Show script output if we just ran a script
        if self.state.last_command and self.state.last_command.startswith("RUN_SCRIPT:"):
            if self.state.script_output:
                state_parts.append(f"\nScript Output:\n{self.state.script_output}")
            
        return "\n".join(state_parts)

    def validate_response(self, response: str) -> Dict[str, Any]:
        """Validate agent response and extract commands.
        
        Args:
            response: Agent's response string
            
        Returns:
            Validation result with valid flag and optional message
        """
        # Check for repository commands first
        ls_match = self.LS_PATTERN.search(response)
        if ls_match:
            return {
                "valid": True,
                "action": "ls",
                "path": ls_match.group(1).strip()
            }
            
        # Check for script command
        if "RUN_SCRIPT:" in response:
            # Direct file execution case
            if response.strip().startswith("RUN_SCRIPT: ") and response.strip().endswith(".py"):
                script_file = response.replace("RUN_SCRIPT:", "").strip()
                return {
                    "valid": True,
                    "action": "script",
                    "script": script_file
                }
            
            # Code block case
            if "```python" in response:
                try:
                    code_start = response.index("```python") + 9
                    code_end = response.index("```", code_start)
                    code = response[code_start:code_end].strip()
                    if code:
                        return {
                            "valid": True,
                            "action": "script",
                            "script": code
                        }
                except ValueError:
                    pass
                    
            return {
                "valid": False,
                "message": "Script command must be either a .py filename or a Python code block: RUN_SCRIPT: ```python\\ncode\\n```"
            }
            
        read_match = self.READ_PATTERN.search(response)
        if read_match:
            return {
                "valid": True,
                "action": "read",
                "path": read_match.group(1).strip(),
                "version": int(read_match.group(2)) if read_match.group(2) else None
            }
            
        write_match = self.WRITE_PATTERN.search(response)
        if write_match:
            return {
                "valid": True,
                "action": "write",
                "path": write_match.group(1).strip(),
                "language": write_match.group(2),
                "content": write_match.group(3)
            }
            
        diff_match = self.DIFF_PATTERN.search(response)
        if diff_match:
            return {
                "valid": True,
                "action": "diff",
                "path": diff_match.group(1).strip(),
                "version1": int(diff_match.group(2)) if diff_match.group(2) else None,
                "version2": int(diff_match.group(3)) if diff_match.group(3) else None
            }
            
        history_match = self.HISTORY_PATTERN.search(response)
        if history_match:
            return {
                "valid": True,
                "action": "history",
                "path": history_match.group(1).strip()
            }
            
        logs_match = self.LOGS_PATTERN.search(response)
        if logs_match:
            return {
                "valid": True,
                "action": "logs",
                "limit": int(logs_match.group(1)) if logs_match.group(1) else None
            }
            
        # Check for search command
        search_match = self.SEARCH_PATTERN.search(response)
        if search_match:
            return {
                "valid": True,
                "action": "search",
                "query": search_match.group(1).strip()
            }
            
        # Check for read_url command
        read_url_match = self.READ_URL_PATTERN.search(response)
        if read_url_match:
            url = read_url_match.group(1).strip()
            if not url.startswith(('http://', 'https://')):
                return {
                    "valid": False,
                    "message": "Invalid URL format. URL must start with http:// or https://"
                }
            return {
                "valid": True,
                "action": "read_url",
                "url": url
            }
            
        # Check for create_note command
        create_note_match = self.CREATE_NOTE_PATTERN.search(response)
        if create_note_match:
            try:
                # Parse the JSON content
                note_data = json.loads(create_note_match.group(1))
                
                # Validate required fields
                if "title" not in note_data or "text" not in note_data:
                    return {
                        "valid": False,
                        "message": "Note must contain both 'title' and 'text' fields"
                    }
                    
                if not note_data["title"].strip() or not note_data["text"].strip():
                    return {
                        "valid": False,
                        "message": "Note title and text cannot be empty"
                    }
                    
                return {
                    "valid": True,
                    "action": "create_note",
                    "title": note_data["title"],
                    "text": note_data["text"]
                }
                
            except json.JSONDecodeError:
                return {
                    "valid": False,
                    "message": "Invalid JSON format. Use: CREATE_NOTE: ```json\\n{\"title\": \"Your Title\", \"text\": \"Your Text\"}\\n```"
                }
                
        # Check for read_notes command
        read_notes_match = self.READ_NOTES_PATTERN.search(response)
        if read_notes_match:
            title = read_notes_match.group(1).strip()
            if not title:
                return {
                    "valid": False,
                    "message": "Note title cannot be empty"
                }
            return {
                "valid": True,
                "action": "read_notes",
                "title": title
            }
            
        # Check for read_paper command
        read_paper_match = self.READ_PAPER_PATTERN.search(response)
        if read_paper_match:
            return {
                "valid": True,
                "action": "read_paper"
            }
            
        # Check for edit_paper command
        edit_paper_match = self.EDIT_PAPER_PATTERN.search(response)
        if edit_paper_match:
            diff_content = edit_paper_match.group(1)
            if not diff_content:
                return {
                    "valid": False,
                    "message": "Diff content cannot be empty"
                }
            return {
                "valid": True,
                "action": "edit_paper",
                "diff": diff_content
            }
            
        # Check for ask_question command
        ask_question_match = self.ASK_QUESTION_PATTERN.search(response)
        if ask_question_match:
            question = ask_question_match.group(1).strip()
            if not question:
                return {
                    "valid": False,
                    "message": "Question cannot be empty"
                }
            return {
                "valid": True,
                "action": "ask_question",
                "question": question
            }
            
        # Analyze what went wrong with the response
        error_details = []
        if "SEARCH:" in response:
            error_details.append("SEARCH command found but in incorrect format. Use exactly: SEARCH: your query here")
        if "READ_URL:" in response:
            error_details.append("READ_URL command found but in incorrect format. Use exactly: READ_URL: https://example.com")
        if "CREATE_NOTE:" in response:
            error_details.append("CREATE_NOTE command found but in incorrect format. Use exactly: CREATE_NOTE: ```json\\n{\"title\": \"Your Title\", \"text\": \"Your Text\"}\\n```")
        if "READ_NOTES:" in response:
            error_details.append("READ_NOTES command found but in incorrect format. Use exactly: READ_NOTES: note_title")
        if "READ_MY_PAPER" in response:
            error_details.append("READ_MY_PAPER command found but in incorrect format. Use exactly: READ_MY_PAPER")
        if "EDIT_MY_PAPER:" in response:
            error_details.append("EDIT_MY_PAPER command found but in incorrect format. Use exactly: EDIT_MY_PAPER: ```diff\\n- old line\\n+ new line\\n```")
        if "RUN_SCRIPT:" in response or "```python" in response:
            error_details.append("Script command found but in incorrect format. Use exactly: RUN_SCRIPT: ```python\\ncode\\n```")
        if "ASK_CLARIFYING_QUESTION:" in response:
            error_details.append("ASK_CLARIFYING_QUESTION command found but in incorrect format. Use exactly: ASK_CLARIFYING_QUESTION: your question here")
            
        error_msg = "Invalid command format. Please use one of: SEARCH, READ_URL, CREATE_NOTE, READ_NOTES, READ_MY_PAPER, EDIT_MY_PAPER, RUN_SCRIPT, or ASK_CLARIFYING_QUESTION"
        if error_details:
            error_msg += "\n\nSpecific issues found:\n- " + "\n- ".join(error_details)
        
        return {
            "valid": False,
            "message": error_msg
        }

    def ask_question(self, question: str) -> str:
        """Ask a question to the user and get their response.
        Uses the mentor LLMPlayer to maintain chat history and provide consistent feedback.
        
        Args:
            question: The question to ask
            
        Returns:
            Mentor's response with enhanced feedback
        """
        # Get current context
        paper_content = self.read_paper()
        repo_tree = self.ls_repository(".")
        
        # Get most recent note if any exist
        latest_note = ""
        if self.state.notes:
            latest_title = list(self.state.notes.keys())[-1]
            latest_note = self.read_note(latest_title)

        # Include last script error if any
        script_error = ""
        if hasattr(self.state, 'last_script_error') and self.state.last_script_error:
            script_error = f"\nLAST SCRIPT ERROR:\n{self.state.last_script_error}"

        # Send context and question to mentor
        mentor_prompt = f"""Current Research Context:

PAPER CONTENT:
{paper_content}

REPOSITORY STRUCTURE:
{repo_tree}

LATEST RESEARCH NOTE:
{latest_note}{script_error}

STUDENT QUESTION:
{question}

Please provide harsh but constructive criticism and feedback to help improve the research.
Guide them towards something that would be useful! Not esoteric, but useful and incredible research."""

        # Get mentor's response using LLMPlayer (maintains chat history)
        mentor_response = self.mentor_player.get_response({
            "role": "user",
            "content": mentor_prompt
        })

        # Print interaction for user
        print(f"\nQuestion from Research Agent: {question}")
        print("\nAI Mentor Feedback:")
        print(mentor_response)
        print("\nPlease provide your response:")
        
        return mentor_response

    def run(self, players: List[Any]) -> Dict[str, Any]:
        """Run the research game with provided players.
        
        Args:
            players: List of player instances
            
        Returns:
            Game results dictionary
            
        Raises:
            ValueError: If no players provided
            OSError: If virtual environment setup fails
        """
        if not players:
            raise ValueError("Need at least one player")
            
        agent = players[0]
        
        print("\n=== Starting Research Game ===")
        
        # Setup virtual environment if it doesn't exist
        venv_python = self.venv_dir / "bin" / "python"
        if not venv_python.exists():
            try:
                print("\nSetting up virtual environment...")
                subprocess.run(
                    [sys.executable, "-m", "venv", str(self.venv_dir)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Install requirements if requirements.txt exists
                requirements_file = self.config.run_dir / "requirements.txt"
                if requirements_file.exists():
                    print("Installing requirements...")
                    venv_pip = self.venv_dir / "bin" / "pip"
                    subprocess.run(
                        [str(venv_pip), "install", "-r", str(requirements_file)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to setup virtual environment: {e.stderr}"
                print(f"Error: {error_msg}")
                print(f"Expected venv at: {self.venv_dir}")
                raise OSError(error_msg) from e
            except Exception as e:
                error_msg = f"Unexpected error setting up virtual environment: {str(e)}"
                print(f"Error: {error_msg}")
                print(f"Expected venv at: {self.venv_dir}")
                raise
        
        # Verify repository structure
        try:
            print("\nVerifying repository structure...")
            self.repo.ls(".")
            print("Repository structure verified")
        except Exception as e:
            error_msg = f"Repository structure check failed: {str(e)}"
            print(f"\nError: {error_msg}")
            print(f"Repository directory: {self.repo_dir}")
            raise OSError(error_msg)
            
        print("\nEnvironment check passed - starting game loop")
        
        while self.state.turn_count < self.config.max_turns:
            print(f"\n--- Turn {self.state.turn_count + 1} ---")
            
            # Get current state
            state_message = {
                "role": "user",
                "content": self.get_current_state()
            }
            
            try:
                # Get player action
                print("\nWaiting for agent response...")
                response = agent.get_response(state_message)
                print(f"\nAgent response:\n{response}")
                
                validation = self.validate_response(response)
                
                if not validation["valid"]:
                    error_msg = validation["message"]
                    print(f"\nInvalid response: {error_msg}")
                    # Add error message as system message to guide the model
                    agent.add_message({
                        "role": "system",
                        "content": f"Your last response did not contain a valid command. {error_msg}\n\nPlease try again with a properly formatted command."
                    })
                    continue
                
                # Handle different actions
                action_type = validation["action"]
                
                if action_type == "ls":
                    print(f"\nListing repository contents: {validation['path']}")
                    result = self.ls_repository(validation["path"])
                    print(result)
                    self.state.last_command = f"LS: {validation['path']}"
                    
                elif action_type == "read":
                    print(f"\nReading file: {validation['path']}")
                    if validation.get("version"):
                        print(f"Version: {validation['version']}")
                    content = self.read_repository_file(validation["path"], validation.get("version"))
                    print(content)
                    self.state.last_command = f"READ: {validation['path']}"
                    
                elif action_type == "write":
                    print(f"\nWriting file: {validation['path']}")
                    result = self.write_repository_file(validation["path"], validation["content"])
                    print(result)
                    self.state.last_command = f"WRITE: {validation['path']}"
                    
                elif action_type == "diff":
                    print(f"\nGetting diff for: {validation['path']}")
                    if validation.get("version1"):
                        print(f"Between version {validation['version1']} ", end="")
                        if validation.get("version2"):
                            print(f"and version {validation['version2']}")
                        else:
                            print("and current")
                    diff = self.diff_repository_file(
                        validation["path"],
                        validation.get("version1"),
                        validation.get("version2")
                    )
                    print(diff)
                    self.state.last_command = f"DIFF: {validation['path']}"
                    
                elif action_type == "history":
                    print(f"\nGetting history for: {validation['path']}")
                    history = self.get_repository_history(validation["path"])
                    print(history)
                    self.state.last_command = f"HISTORY: {validation['path']}"
                    
                elif action_type == "logs":
                    print("\nGetting repository logs")
                    if validation.get("limit"):
                        print(f"Limit: {validation['limit']} entries")
                    logs = self.get_repository_logs(validation.get("limit"))
                    print(logs)
                    self.state.last_command = "LOGS"
                    
                elif action_type == "search":
                    print(f"\nSearching papers: {validation['query']}")
                    results = self.search(validation["query"])
                    self.state.search_results = [results]
                    # Print the actual search results
                    print("\nSearch Results:")
                    print(results)
                    self.state.last_command = f"SEARCH: {validation['query']}"
                    
                elif action_type == "read_url":
                    print(f"\nReading webpage: {validation['url']}")
                    webpage_content = self.read_webpage(validation["url"])
                    self.state.current_paper["content"] = [webpage_content]
                    self.state.last_command = f"READ_URL: {validation['url']}"
                    
                elif action_type == "create_note":
                    print(f"\nCreating note: {validation['title']}")
                    result = self.create_note(validation["title"], validation["text"])
                    print(result)
                    self.state.last_command = f"CREATE_NOTE: {validation['title']}"
                    
                elif action_type == "read_notes":
                    print(f"\nReading note: {validation['title']}")
                    note_content = self.read_note(validation["title"])
                    print(note_content)
                    self.state.last_command = f"READ_NOTES: {validation['title']}"
                    
                elif action_type == "read_paper":
                    print("\nReading paper...")
                    paper_content = self.read_paper()
                    print(paper_content)
                    self.state.last_command = "READ_MY_PAPER"
                    
                elif action_type == "edit_paper":
                    print("\nEditing paper...")
                    result = self.edit_paper(validation["diff"])
                    print(result)
                    self.state.last_command = f"EDIT_MY_PAPER: {validation['diff']}"
                    
                elif action_type == "script":
                    print("\nExecuting script...")
                    self.state.script_output = self.run_script(validation["script"])
                    print(f"\nScript output:\n{self.state.script_output}")
                    self.state.last_command = f"RUN_SCRIPT: {validation['script']}"
                    
                elif action_type == "ask_question":
                    print(f"\nAsking question: {validation['question']}")
                    response = self.ask_question(validation['question'])
                    # Add the question and response to the agent's messages
                    agent.add_message({
                        "role": "user",
                        "content": f"Your question: {validation['question']}\nMentor response: {response}"
                    })
                    self.state.last_command = f"ASK_CLARIFYING_QUESTION: {validation['question']}"
                
                self.state.turn_count += 1
                
            except Exception as e:
                print(f"\nError during turn: {str(e)}")
                continue
        
        self.end_time = datetime.now().isoformat()
        print("\n=== Game Complete ===")
        
        return {
            "status": "complete",
            "turns": self.state.turn_count,
            "final_paper": self.read_paper(),
            "start_time": self.start_time,
            "end_time": self.end_time
        }

    def ls_repository(self, path: str) -> str:
        """List contents of repository directory.
        
        Args:
            path: Directory path to list
            
        Returns:
            Formatted directory listing
        """
        try:
            entries = self.repo.ls(path)
            
            # Format entries
            lines = []
            for entry in entries:
                type_char = 'd' if entry['type'] == 'directory' else '-'
                size = entry['size']
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}K"
                else:
                    size_str = f"{size/(1024*1024):.1f}M"
                    
                version_str = f" [{entry.get('versions', '-')}v]" if entry['type'] == 'file' else ""
                lines.append(f"{type_char} {size_str:>8} {entry['modified']} {entry['name']}{version_str}")
                
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def read_repository_file(self, path: str, version: Optional[int] = None) -> str:
        """Read file from repository.
        
        Args:
            path: File path to read
            version: Optional version to read
            
        Returns:
            File contents
        """
        try:
            return self.repo.read_file(path, version)
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_repository_file(self, path: str, content: str) -> str:
        """Write file to repository.
        
        Args:
            path: File path to write
            content: Content to write
            
        Returns:
            Status message
        """
        try:
            version = self.repo.write_file(path, content)
            return f"File written successfully as version {version}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def diff_repository_file(self, path: str, version1: Optional[int], version2: Optional[int] = None) -> str:
        """Get diff between file versions.
        
        Args:
            path: File path
            version1: First version
            version2: Second version (defaults to current)
            
        Returns:
            Diff output
        """
        try:
            return self.repo.diff(path, version1, version2)
        except Exception as e:
            return f"Error getting diff: {str(e)}"

    def get_repository_history(self, path: str) -> str:
        """Get file version history.
        
        Args:
            path: File path
            
        Returns:
            Formatted history
        """
        try:
            versions = self.repo.get_history(path)
            lines = []
            for v in versions:
                lines.append(f"Version {v['version']}: {v['timestamp']} [{v['hash'][:8]}]")
            return "\n".join(lines)
        except Exception as e:
            return f"Error getting history: {str(e)}"

    def get_repository_logs(self, limit: Optional[int] = None) -> str:
        """Get repository operation logs.
        
        Args:
            limit: Optional maximum number of entries
            
        Returns:
            Formatted log entries
        """
        try:
            logs = self.repo.get_logs(limit or 100)
            lines = []
            for log in logs:
                lines.append(f"{log['timestamp']} {log['operation']} {log['path']}")
                if log['details']:
                    lines.append(f"  {json.dumps(log['details'])}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error getting logs: {str(e)}"
