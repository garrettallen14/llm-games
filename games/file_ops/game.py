"""File operations game implementation."""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass

from base.llm_player import BaseLLMPlayer

@dataclass
class FileOpsConfig:
    """Configuration settings for file operations game.
    
    Attributes:
        run_dir: Directory for game artifacts
        max_turns: Maximum allowed game turns
    """
    run_dir: Path
    max_turns: int = 100

    def __post_init__(self):
        """Setup repository and venv directories after initialization."""
        self.repository_dir = self.run_dir / "repository"
        self.venv_dir = self.repository_dir / "venv"
        
        # Create directories
        self.repository_dir.mkdir(parents=True, exist_ok=True)
        
        # Create virtual environment
        os.system(f"python3 -m venv {self.venv_dir}")

class FileOpsGame:
    """File operations game manager."""
    
    def __init__(self, run_dir: Path, max_turns: int = 100) -> None:
        """Initialize file operations game with configuration.

        Args:
            run_dir: Directory for game artifacts
            max_turns: Maximum allowed game turns
        """
        self.config = FileOpsConfig(run_dir=run_dir, max_turns=max_turns)
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None

    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for the agent."""
        return {
            "role": "system",
            "content": f"""You are an autonomous AI agent with complete creative freedom and technical expertise. Your mission is to independently design, architect, and implement sophisticated software solutions that meet the highest standards of quality and innovation.

Core Development Philosophy:
1. Complete Creative Independence
   - Make all technical decisions autonomously
   - Design innovative solutions without external guidance
   - Take full ownership of the architecture and implementation

2. Engineering Excellence
   - Write clean, maintainable, and efficient code
   - Follow industry best practices and design patterns
   - Create comprehensive documentation
   - Implement thorough testing strategies

3. Project Management
   - Break down complex requirements into manageable tasks
   - Plan and execute development in logical phases
   - Track progress and maintain project momentum
   - Adapt and refine approach based on results

Available Development Tools:

FILE OPERATIONS:
create_file    {{"path": str, "content": str}}     # Create new project files
read_file      {{"path": str}}                     # Examine existing code
edit_file      {{"path": str, "content": str}}     # Modify implementation
delete_file    {{"path": str}}                     # Remove obsolete files

DIRECTORY OPERATIONS:
create_dir     {{"path": str}}                     # Establish project structure
list_dir       {{"path": str}}                     # Review organization
delete_dir     {{"path": str}}                     # Clean up directories
move           {{"src": str, "dst": str}}          # Reorganize as needed

EXECUTION & PACKAGE MANAGEMENT:
run_script     {{"path": str}}                     # Execute and validate code
pip_install    {{"package": str}}                  # Manage dependencies

USER INTERACTION:
request_user_feedback {{"message": str, "type": str}}  # Request guidance when needed
                                                      # Types: review, next_steps, clarification, ideas

Universal Best Practices:
1. Code Organization
   - Logical project structure
   - Clear module boundaries
   - Consistent naming conventions
   - Proper separation of concerns

2. Code Quality
   - Comprehensive error handling
   - Input validation
   - Performance optimization
   - Security considerations

3. Documentation
   - Clear code comments
   - API documentation
   - Setup instructions
   - Usage examples

4. Testing
   - Unit tests
   - Integration tests
   - Edge case coverage
   - Performance benchmarks

Development Constraints:
1. All code stays within repository directory
2. Use virtual environment for dependencies
3. Follow clean code principles
4. One action per turn

[USER PROJECT SPECIFICATION GOES HERE]
> Please create a repository where you have many Agents with tools to work on a repository, just like you do.
>
> What I would really like is to have the Agents be able to communicate with each other as well! So they can organize and work on the right parts of the correct directories and everyone can communicate properly.
> Maybe you can think of this as you spawning in a bunch of Agents that you can speak with and then you can all start working on the repository that you currently have!
> I don't want to influence you too much, but keep it simple to begin with and start by creating the initial communication protocol and then spawn in Agents that can help you code and have message history just like you!
> NOTE: I will also provide an OpenRouter .env into your repository for you to use OpenRouter with your Agents!
> OR uses OpenAI llm calling framework but with: from openai import OpenAI

from openai import OpenAI
client = OpenAI(
  base_url='https://openrouter.ai/api/v1',
  api_key='from the .env file I gave you',
)
completion = client.chat.completions.create(
  model='openai/o3-mini',
  messages=[
    {{
      'role': 'user',
      'content': 'What is the meaning of life?'
    }}
  ]
)
print(completion.choices[0].message.content)


IMPORTANT: You can only submit one action per turn. Use them wisely to progressively build your solution.
"""
        }

    def _get_turn_refresher(self, turn: int) -> str:
        """Provide a concise refresher of current state."""
        # Get repository contents
        try:
            repo_contents = os.listdir(self.config.repository_dir)
            contents = repo_contents if repo_contents else ["(empty)"]
        except Exception:
            contents = ["(error listing contents)"]
        
        # Construct brief refresher
        refresher = f"""Turn {turn + 1}: Repository Contents: {', '.join(contents)}
Actions: create_file, read_file, edit_file, delete_file, 
         create_dir, list_dir, delete_dir, move, 
         run_script, pip_install, request_user_feedback

Next action?"""
        return refresher

    def run(self, players: List[BaseLLMPlayer]) -> None:
        """Run the file operations game.

        Args:
            players: List of LLM players
        """
        if not players:
            raise ValueError("No players provided")
            
        player = players[0]  # Use the first player
        
        # Initialize the player with system prompt
        player.initialize_with_prompt(self.get_system_prompt())
        
        turn = 0
        while turn < self.config.max_turns:
            # Get current state with refresher
            state_message = {
                "role": "user", 
                "content": self._get_turn_refresher(turn)
            }
            
            # Get response from player
            response = player.get_response(state_message)
            
            # Parse and execute tool
            tool_info = self._parse_tool_response(response)
            if not tool_info:
                message = "Invalid tool format. Please use TOOL: and PARAMS: format."
                print(f"\n[SYSTEM] {message}")
                player.add_message({
                    "role": "system",
                    "content": message
                })
                continue
                
            try:
                result = self._execute_tool(tool_info)
                message = f"Action completed: {result}"
                print(f"\n[SYSTEM] {message}")
                player.add_message({
                    "role": "system",
                    "content": message
                })
            except Exception as e:
                message = f"Error executing tool: {str(e)}"
                print(f"\n[SYSTEM] {message}")
                player.add_message({
                    "role": "system",
                    "content": message
                })
                continue
                
            turn += 1
            
        self.end_time = datetime.now().isoformat()

    def _parse_tool_response(self, response: str) -> Optional[Dict]:
        """Parse the tool response from the LLM."""
        try:
            lines = response.strip().split('\n')
            tool_line = None
            params_line = None
            
            for line in lines:
                if line.startswith('TOOL:'):
                    tool_line = line[5:].strip()
                elif line.startswith('PARAMS:'):
                    params_line = line[7:].strip()
            
            if not tool_line:
                return None
                
            params = {}
            if params_line:
                try:
                    params = eval(params_line)
                except:
                    return None
                    
            return {
                "tool": tool_line,
                "params": params
            }
            
        except Exception as e:
            print(f"Error parsing tool response: {e}")
            return None

    def _execute_tool(self, tool_info: Dict) -> str:
        """Execute the specified file operation tool."""
        tool = tool_info["tool"]
        params = tool_info["params"]
        
        # Ensure all paths are within repository directory
        if "path" in params:
            full_path = self._safe_path(params["path"])
        if "src" in params:
            params["src"] = self._safe_path(params["src"])
        if "dst" in params:
            params["dst"] = self._safe_path(params["dst"])
            
        if tool == "create_file":
            with open(full_path, 'w') as f:
                f.write(params["content"])
            return f"Created file: {params['path']}"
            
        elif tool == "read_file":
            with open(full_path, 'r') as f:
                content = f.read()
            return f"File contents:\n{content}"
            
        elif tool == "edit_file":
            with open(full_path, 'w') as f:
                f.write(params["content"])
            return f"Edited file: {params['path']}"
            
        elif tool == "delete_file":
            os.remove(full_path)
            return f"Deleted file: {params['path']}"
            
        elif tool == "create_dir":
            os.makedirs(full_path, exist_ok=True)
            return f"Created directory: {params['path']}"
            
        elif tool == "list_dir":
            contents = os.listdir(full_path)
            return f"Directory contents:\n{contents}"
            
        elif tool == "delete_dir":
            shutil.rmtree(full_path)
            return f"Deleted directory: {params['path']}"
            
        elif tool == "move":
            shutil.move(params["src"], params["dst"])
            return f"Moved {params['src']} to {params['dst']}"

        elif tool == "run_script":
            venv_python = self.config.venv_dir / "bin" / "python"
            import subprocess
            try:
                # Use subprocess to capture both stdout and stderr
                result = subprocess.run(
                    [str(venv_python), str(full_path)], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                return f"Script output:\n{result.stdout}\n\nScript error output:\n{result.stderr}"
            except subprocess.CalledProcessError as e:
                # If the script fails, return full error details
                return f"""Script Execution Failed:
STDOUT:
{e.stdout}

STDERR:
{e.stderr}

FULL TRACEBACK:
{e.stderr}

Exit Code: {e.returncode}"""

        elif tool == "pip_install":
            venv_pip = self.config.venv_dir / "bin" / "pip"
            package = params["package"]
            result = os.popen(f"{venv_pip} install {package}").read()
            return f"Pip install output:\n{result}"

        elif tool == "request_user_feedback":
            message = params["message"]
            feedback_type = params["type"]
            print(f"[USER FEEDBACK REQUESTED - {feedback_type.upper()}]\n{message}")
            user_response = input("Please provide your feedback: ")
            return f"User Feedback Received: {user_response}"
            
        else:
            raise ValueError(f"Unknown tool: {tool}")

    def _safe_path(self, path: str) -> Path:
        """Ensure path is within repository directory.
        
        Handles various path formats:
        - Absolute paths
        - Relative paths
        - Paths with ./ or ../
        - Paths with multiple slashes
        """
        # Normalize the path, removing redundant separators and resolving ./ and ../
        normalized_path = str(Path(path).resolve())
        repo_path = str(self.config.repository_dir.resolve())
        
        # Check if normalized path is within repository
        if not normalized_path.startswith(repo_path):
            # If not, try to create a path relative to repository
            full_path = (self.config.repository_dir / path).resolve()
            
            # Re-check if this new path is within repository
            if not str(full_path).startswith(repo_path):
                raise ValueError(f"Path {path} is outside repository directory")
            
            return full_path
        
        return Path(normalized_path)
