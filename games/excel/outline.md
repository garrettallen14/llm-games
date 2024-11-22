# Excel Manipulation Game Outline

## Overview
A game where LLMs learn to manipulate Excel spreadsheets through a series of increasingly complex challenges. The game will test the model's ability to understand spreadsheet operations, data manipulation, and formula creation.

## Core Primitive Functions

### 1. Cell Operations
- READ_CELL(sheet_name, cell_reference) -> value
- WRITE_CELL(sheet_name, cell_reference, value)
- GET_CELL_FORMAT(sheet_name, cell_reference) -> format_info
- SET_CELL_FORMAT(sheet_name, cell_reference, format_info)

### 2. Range Operations
- READ_RANGE(sheet_name, start_cell, end_cell) -> values[][]
- WRITE_RANGE(sheet_name, start_cell, end_cell, values[][])
- CLEAR_RANGE(sheet_name, start_cell, end_cell)
- COPY_RANGE(source_sheet, source_range, target_sheet, target_range)

### 3. Formula Operations
- SET_FORMULA(sheet_name, cell_reference, formula)
- GET_FORMULA(sheet_name, cell_reference) -> formula
- EVALUATE_FORMULA(formula, context) -> result

### 4. Sheet Operations
- CREATE_SHEET(sheet_name)
- DELETE_SHEET(sheet_name)
- LIST_SHEETS() -> sheet_names[]
- ACTIVATE_SHEET(sheet_name)

### 5. Formatting Operations
- SET_COLUMN_WIDTH(sheet_name, column, width)
- SET_ROW_HEIGHT(sheet_name, row, height)
- APPLY_STYLE(sheet_name, range, style_info)
- ADD_CONDITIONAL_FORMAT(sheet_name, range, condition, format)

## Challenge Format (JSONL)

Each challenge in the challenges.jsonl file will have the following structure:

```json
{
  "id": "string",                    // Unique identifier for the challenge
  "difficulty": "beginner|intermediate|advanced",
  "category": "string",             // e.g., "formatting", "formulas", "data_analysis"
  "description": "string",          // Human-readable description of the task
  "initial_state": {                // Initial spreadsheet state
    "sheets": [
      {
        "name": "string",
        "data": [[]],              // 2D array of cell values
        "formulas": {},            // Map of cell references to formulas
        "formats": {}              // Map of cell references to format information
      }
    ]
  },
  "expected_state": {              // Expected final state after manipulation
    "sheets": [/*...*/]
  },
  "constraints": {                 // Optional constraints on solution
    "max_operations": number,
    "forbidden_operations": ["operation_names"],
    "required_operations": ["operation_names"]
  },
  "hints": ["string"],            // Optional hints for solving the challenge
  "solution": {                   // Example solution
    "operations": [
      {
        "operation": "string",    // Name of primitive operation
        "params": {}             // Operation parameters
      }
    ]
  }
}
```

## Example Challenges

### 1. Basic Cell Manipulation
```json
{
  "id": "basic_001",
  "difficulty": "beginner",
  "category": "cell_operations",
  "description": "Copy the value from cell A1 to cell B1 and format it as bold",
  "initial_state": {
    "sheets": [{
      "name": "Sheet1",
      "data": [["Hello", ""]]
    }]
  },
  "expected_state": {
    "sheets": [{
      "name": "Sheet1",
      "data": [["Hello", "Hello"]],
      "formats": {
        "B1": {"bold": true}
      }
    }]
  }
}
```

### 2. Formula Creation
```json
{
  "id": "formula_001",
  "difficulty": "intermediate",
  "category": "formulas",
  "description": "Create a sum formula in cell C1 that adds values from A1 and B1",
  "initial_state": {
    "sheets": [{
      "name": "Sheet1",
      "data": [[10, 20, ""]]
    }]
  },
  "expected_state": {
    "sheets": [{
      "name": "Sheet1",
      "data": [[10, 20, 30]],
      "formulas": {
        "C1": "=SUM(A1:B1)"
      }
    }]
  }
}
```

## Learning Mechanisms

1. **Progressive Difficulty**
   - Challenges are organized by difficulty level
   - Models must master basic operations before advancing
   - Performance on previous challenges influences strategy

2. **Context Window**
   - Models can see previous successful solutions
   - Learn from past approaches and patterns
   - Develop more efficient strategies over time

3. **Feedback Loop**
   - Each operation provides immediate feedback
   - Models can see the state after each operation
   - Learn from mistakes and unexpected outcomes

4. **Pattern Recognition**
   - Similar challenges with variations
   - Models learn to recognize common patterns
   - Develop reusable strategies

## Evaluation Metrics

1. **Success Rate**
   - Percentage of challenges completed successfully
   - Success within constraints
   - First-attempt success rate

2. **Efficiency**
   - Number of operations used vs. optimal solution
   - Time taken to complete challenge
   - Resource usage (memory, API calls)

3. **Strategy Quality**
   - Use of advanced features
   - Code reusability
   - Error handling

4. **Learning Progress**
   - Improvement over time
   - Adaptation to new patterns
   - Knowledge transfer between similar challenges
