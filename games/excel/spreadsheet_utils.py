import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def create_random_spreadsheet(rows=9, cols=9, min_val=1, max_val=9, empty_prob=0.4):
    """Create a random spreadsheet with numbers and empty cells"""
    # Create empty DataFrame
    df = pd.DataFrame(
        "",  # Initialize with empty strings
        index=range(rows),
        columns=[chr(65 + i) for i in range(cols)]  # A, B, C, etc.
    )
    
    # Randomly fill cells with numbers or leave empty
    for i in range(rows):
        for j in range(cols):
            if np.random.random() > empty_prob:  # 40% chance of being empty
                df.iloc[i, j] = np.random.randint(min_val, max_val + 1)
    
    return df

def sort_numbers_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort numbers into their corresponding columns (A=1, B=2, etc) and add row sums"""
    result = df.copy()
    
    # Clear all cells first
    for col in result.columns:
        result[col] = ""
    
    # Collect all numbers from original DataFrame
    numbers = []
    for i in range(len(df)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            if val != "" and not pd.isna(val):
                numbers.append(int(val))
    
    # Sort numbers into appropriate columns
    for num in sorted(numbers):
        if 1 <= num <= 9:  # Only process numbers 1-9
            col = chr(64 + num)  # A=1, B=2, etc.
            # Find first empty cell in the column
            for row in range(len(result)):
                if result.loc[row, col] == "":
                    result.loc[row, col] = num
                    break
    
    # Add column for sums (column J)
    result['J'] = ""  # Initialize sum column
    
    # Calculate row sums (excluding empty cells and sum column)
    for i in range(len(result)):
        row_values = [float(x) for x in result.iloc[i, :-1] if x != ""]
        if row_values:  # Only add sum if there are numbers in the row
            result.iloc[i, -1] = int(sum(row_values))
    
    return result

def render_spreadsheet(df, cell_width=60, cell_height=30, last_move=None):
    """Render a pandas DataFrame as an Excel-like image
    
    Args:
        df: DataFrame to render
        cell_width: Width of each cell
        cell_height: Height of each cell
        last_move: Tuple of (source_cell, target_cell) to highlight
    """
    # Add row and column headers
    df_with_headers = df.copy()
    
    # Calculate dimensions
    n_rows, n_cols = df_with_headers.shape
    width = (n_cols + 1) * cell_width
    height = (n_rows + 1) * cell_height
    
    # Create image with a light blue background
    image = Image.new('RGB', (width, height), '#F0F8FF')  # Light blue background
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
        header_font = ImageFont.truetype("Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()
    
    # Draw grid with improved styling
    for i in range(n_rows + 2):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill='#A9A9A9', width=2)  # Darker grid lines
    for i in range(n_cols + 2):
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill='#A9A9A9', width=2)
    
    # Draw column headers with improved styling
    for i in range(n_cols):
        col_letter = chr(65 + i)
        x = (i + 1) * cell_width + cell_width//3
        y = cell_height//3
        draw.text((x, y), col_letter, fill='#000080', font=header_font)  # Navy blue headers
    
    # Draw row headers
    for i in range(n_rows):
        x = cell_width//3
        y = (i + 1) * cell_height + cell_height//3
        draw.text((x, y), str(i+1), fill='#000080', font=header_font)
    
    # Helper function to get cell coordinates
    def get_cell_rect(col, row):
        x1 = (col + 1) * cell_width
        y1 = (row + 1) * cell_height
        return (x1, y1, x1 + cell_width, y1 + cell_height)
    
    # Draw last move highlights if provided
    if last_move:
        source, target = last_move
        source_col = ord(source[0]) - ord('A')
        source_row = int(source[1]) - 1
        target_col = ord(target[0]) - ord('A')
        target_row = int(target[1]) - 1
        
        # Draw source cell background (black)
        source_rect = get_cell_rect(source_col, source_row)
        draw.rectangle(source_rect, fill='#000000')
        
        # Draw target cell background (blue)
        target_rect = get_cell_rect(target_col, target_row)
        draw.rectangle(target_rect, fill='#0000FF')
    
    # Draw values with improved styling
    for i in range(n_rows):
        for j in range(n_cols):
            value = df.iloc[i, j]
            if pd.isna(value) or value == "":
                value = ""
            else:
                try:
                    float_val = float(value)
                    value = str(int(float_val)) if float_val.is_integer() else str(float_val)
                except (ValueError, TypeError):
                    value = str(value)
            
            # Center the text in the cell
            x = (j + 1) * cell_width + cell_width//2
            y = (i + 1) * cell_height + cell_height//2
            
            # Get text size for centering
            text_bbox = draw.textbbox((0, 0), value, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Set text color based on last move
            text_color = '#FFFFFF' if last_move and (
                (j == ord(last_move[0][0]) - ord('A') and i == int(last_move[0][1]) - 1) or
                (j == ord(last_move[1][0]) - ord('A') and i == int(last_move[1][1]) - 1)
            ) else '#000000'
            
            # Draw centered text
            draw.text(
                (x - text_width//2, y - text_height//2),
                value,
                fill=text_color,
                font=font
            )
    
    return image

def print_spreadsheet(df: pd.DataFrame):
    """Print spreadsheet in ASCII format with empty cells as '-'"""
    # Create a copy and replace empty/NaN with '-'
    df_print = df.copy()
    df_print = df_print.fillna('-')
    df_print = df_print.replace('', '-')
    
    # Convert all numbers to integers if they're whole numbers
    for col in df_print.columns:
        df_print[col] = df_print[col].apply(lambda x: int(float(x)) if isinstance(x, (int, float)) and not pd.isna(x) and str(x) != '-' and float(x).is_integer() else x)
    
    # Create a more visually appealing ASCII representation
    output = []
    
    # Add column headers with spacing
    header = "     " + "  ".join(f" {col} " for col in df_print.columns)
    output.append(header)
    output.append("   " + "─" * (len(header) - 3))
    
    # Add rows with row numbers
    for idx, row in df_print.iterrows():
        row_str = f" {idx+1} │ " + "  ".join(f" {str(val):1} " for val in row)
        output.append(row_str)
    
    return "\n".join(output)

if __name__ == "__main__":
    # Create a random spreadsheet
    df = create_random_spreadsheet(9, 9)
    print("\nInitial State:")
    print(print_spreadsheet(df))
    
    # Sort numbers into columns
    sorted_df = sort_numbers_to_columns(df)
    print("\nAfter sorting numbers to columns:")
    print(print_spreadsheet(sorted_df))
    
    # Generate visualizations
    before = render_spreadsheet(df)
    after = render_spreadsheet(sorted_df)
    
    before.save("before.png")
    after.save("after.png")
    print("\nImages saved as 'before.png' and 'after.png'")
