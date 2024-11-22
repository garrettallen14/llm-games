from pathlib import Path
from typing import Dict, Any, List, Optional
from games.excel.underwriting_model import UnderwritingModel
from games.excel.spreadsheet_utils import render_spreadsheet
import pandas as pd
import base64

class ExcelGame:
    """Property underwriting game using Excel-like commands"""
    
    def __init__(self, run_dir: Path, max_turns: int = 50):
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.current_turn = 0
        self.game_over = False
        self.messages: List[Dict[str, str]] = []
        
        # Initialize model
        self.model = UnderwritingModel()
    
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the game's system prompt"""
        return {
            "role": "system",
            "content": """You are a commercial real estate analyst building a DCF model.

PROPERTY DETAILS
---------------
Class B Office Building
Size: 50,000 SF
Current Occupancy: 85%
In-Place Rent: $25/SF
Market Rent: $28/SF
OpEx: $12/SF
Cap Rate: 6.5%

MODEL REQUIREMENTS
----------------
Build a 5-year DCF model showing:
1. Revenue (3% annual growth)
2. Occupancy (stabilize at 92% by year 3)
3. Operating Expenses (2% annual growth)
4. NOI
5. CapEx ($2/SF annually)
6. Free Cash Flow
7. Terminal Value (6.5% exit cap)
8. Discount Rate (8%)

COMMANDS
--------
INSERT: <value> <cell>    # Add number to cell
FORMULA: <cell> <expr>    # Add formula
NPV: <range> <rate_cell>  # Calculate NPV
DELETE: <cell>           # Clear cell
TYPE: <cell> <type>      # Set cell type

TEMPLATE LAYOUT
-------------
A0-A5: Years (0-5)
B0-B5: Revenue
C0-C5: Occupancy
D0-D5: Effective Revenue
E0-E5: OpEx
F0-F5: NOI
G0-G5: CapEx
H0-H5: Free Cash Flow
I5: Terminal Value
J0-J5: Total Cash Flow
K0-K5: Discount Factors
L0-L5: PV of Cash Flow
M0: Final Property Value

Your task is complete when you have:
1. Populated all required cells
2. Verified all calculations
3. Calculated final property value in M0

Make your moves one at a time using the exact command format shown above.
"""
        }
    
    def _model_to_dataframe(self) -> pd.DataFrame:
        """Convert model to DataFrame for rendering"""
        # Create empty DataFrame with 20 rows
        cols = [chr(65 + i) for i in range(13)]  # A through M
        df = pd.DataFrame(
            "",
            index=range(20),  # Increased to 20 rows
            columns=cols
        )
        
        # Fill values
        for cell, value in self.model.cells.items():
            col = cell[0]
            row = int(cell[1:])
            if row < 20:  # Only fill if within our display range
                df.iloc[row, ord(col) - ord('A')] = value.value
        
        # Add type hints in empty cells
        type_hints = {
            'A': 'TYPE: Year',
            'B': 'TYPE: Revenue ($)',
            'C': 'TYPE: Occupancy (%)',
            'D': 'TYPE: =B*C',
            'E': 'TYPE: OpEx ($)',
            'F': 'TYPE: =D-E',
            'G': 'TYPE: CapEx ($)',
            'H': 'TYPE: =F-G',
            'I': 'TYPE: Terminal Value',
            'J': 'TYPE: Cash Flow',
            'K': 'TYPE: =1/(1+r)^n',
            'L': 'TYPE: =J*K',
            'M': 'TYPE: Final Value'
        }
        
        for col in df.columns:
            mask = df[col] == ""
            df.loc[mask, col] = type_hints[col]
        
        return df
    
    def get_state(self) -> Dict[str, Any]:
        """Get current game state"""
        df = self._model_to_dataframe()
        board_ascii = df.to_string()
        
        return {
            "board_ascii": board_ascii,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns
        }
    
    def get_board_image(self) -> Optional[str]:
        """Get base64 encoded PNG of current board state"""
        try:
            df = self._model_to_dataframe()
            
            # Calculate cell width based on content
            max_content_width = 0
            for col in df.columns:
                col_content = [str(val) for val in df[col]]
                max_width = max(len(str(val)) for val in col_content)
                max_content_width = max(max_content_width, max_width)
            
            # Set cell width based on content (8 pixels per character plus padding)
            cell_width = max(max_content_width * 8 + 16, 60)  # minimum 60 pixels
            
            # Generate image with calculated width
            img_data = render_spreadsheet(
                df,
                cell_width=cell_width,
                cell_height=30
            )
            
            # Save and encode
            img_path = str("board.png")
            img_data.save(img_path)
            
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            print(f"Warning: Failed to generate board image: {e}")
            return None
    
    def process_command(self, command: str) -> str:
        """Process a command from the player"""
        success, message = self.model.process_command(command)
        return message if success else f"Error: {message}"
    
    def run(self, players: List['BaseLLMPlayer']) -> Dict[str, Any]:
        """Run the game"""
        print("\n🏢 Starting Property Underwriting Game")
        player = players[0]  # Single player game
        
        # Initialize with system prompt
        system_prompt = self.get_system_prompt()
        self.messages.append(system_prompt)
        
        while not self.game_over and self.current_turn < self.max_turns:
            # Get current state
            state = self.get_state()
            
            # Format board state message
            board_state = (
                f"Current model state (Turn {state['current_turn'] + 1}/{state['max_turns']}):\n\n"
                f"{state['board_ascii']}\n\n"
                f"Enter one command (INSERT, FORMULA, NPV, DELETE, or TYPE)"
            )
            
            # Create state message
            state_message = {
                "role": "user",
                "content": board_state
            }
            self.messages.append(state_message)
            
            # Get player's move with board image
            response = player.get_response(state_message, self.get_board_image())
            print(f"\n🤖 Player: {response}")
            
            # Add player's response to messages
            self.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Process commands
            commands = response.split('\n')
            for command in commands:
                if command.strip():
                    result = self.process_command(command.strip())
                    print(f"💬 Game: {result}")
                    
                    # Add result to messages
                    self.messages.append({
                        "role": "system" if not result.startswith("Error") else "user",
                        "content": result
                    })
            
            # Check if model is complete
            success, message = self.model.is_complete()
            if success:
                print("\n🎉 Model Complete!")
                self.game_over = True
                self.messages.append({
                    "role": "system",
                    "content": "🎉 Congratulations! The DCF model is complete and correct!"
                })
                return {"outcome": "success", "turns": self.current_turn + 1}
            
            self.current_turn += 1
        
        if not self.game_over:
            print("\n⏰ Game Over - Max turns reached!")
            self.messages.append({
                "role": "system",
                "content": "⏰ Game Over - Maximum turns reached!"
            })
        
        return {"outcome": "failure", "turns": self.current_turn}
