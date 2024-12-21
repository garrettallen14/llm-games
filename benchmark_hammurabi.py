#!/usr/bin/env python3

import subprocess
import json
import os
from datetime import datetime
from typing import List, Dict
import time
import random

# Models to benchmark
MODELS = [
    # "openai/o1-preview",
    "meta-llama/llama-3.3-70b-instruct",
    "anthropic/claude-3.5-sonnet",
    # "google/gemini-pro-1.5",
    # "qwen/qwq-32b-preview", 
    # "openai/o1-mini", 
    # "anthropic/claude-3.5-haiku",
    # "openai/gpt-4o-mini",
    # "openai/gpt-4o-2024-11-20",
    # "google/gemini-flash-1.5-8b",
    # "meta-llama/llama-3.1-8b-instruct",
    # "mistralai/mistral-7b-instruct",

    # "meta-llama/llama-3.2-3b-instruct",
    # "amazon/nova-lite-v1",
    # "amazon/nova-micro-v1",
    # "amazon/nova-pro-v1",
    # "cohere/command-r-08-2024",
    # "meta-llama/llama-3.1-405b-instruct",
    # "nvidia/llama-3.1-nemotron-70b-instruct",
]

# Number of runs per model
RUNS_PER_MODEL = 10

# Generate a consistent set of 10 seeds for all models
def generate_global_seeds() -> List[int]:
    """Generate a consistent set of 10 global seeds"""
    # Use a fixed base seed for reproducibility
    base_seed = 42  # Chosen arbitrarily but consistently
    random.seed(base_seed)
    
    # Generate 10 unique, deterministic seeds
    seeds = [
        random.randint(0, 2**32 - 1)
        for _ in range(RUNS_PER_MODEL)
    ]
    
    return seeds

def run_game(model: str, run_number: int, seed: int) -> Dict:
    """Run a single game of Hammurabi with the specified model and seed"""
    try:
        print(f"Running {model} (Run {run_number}) with seed {seed}")

        # Capture both stdout and stderr while showing live output
        process = subprocess.Popen(
            [
                "python3",
                "-m",
                "base.runner",
                "--game",
                "hammurabi",
                "--models",
                model,
                "--max_turns",
                "10",
                "--seed",  # Add seed argument
                str(seed)  # Convert seed to string
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Read and print stdout in real-time while storing it
        output_lines = []
        while True:
            # Use readline() to get output line by line
            line = process.stdout.readline()
            if line:
                print(line, end='', flush=True)  # Print immediately
                output_lines.append(line)

            # Check if process has finished
            if process.poll() is not None:
                break

        # Get any remaining output and errors
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout, end='', flush=True)
            output_lines.extend(stdout.splitlines())

        # Find the game result in the output
        for line in reversed(output_lines):
            if "Game complete! Result:" in line:
                # Find the JSON part after "Game complete! Result:"
                json_str = line[line.find("{"):].strip()
                # Convert single quotes to double quotes for valid JSON
                json_str = json_str.replace("'", '"')
                game_result = json.loads(json_str)
                game_result["model"] = model
                game_result["run_number"] = run_number
                game_result["timestamp"] = datetime.now().isoformat()
                return game_result

        # If we get here, we couldn't find a valid result
        return {
            "status": "error",
            "model": model,
            "run_number": run_number,
            "timestamp": datetime.now().isoformat(),
            "error": "Could not find game result in output",
            "stderr": "\n".join(output_lines)
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "model": model,
            "run_number": run_number,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "stderr": e.stderr
        }
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "model": model,
            "run_number": run_number,
            "timestamp": datetime.now().isoformat(),
            "error": f"Failed to parse game result: {str(e)}",
            "stderr": "\n".join(output_lines)
        }
    except Exception as e:
        return {
            "status": "error",
            "model": model,
            "run_number": run_number,
            "timestamp": datetime.now().isoformat(),
            "error": f"Unexpected error: {str(e)}"
        }

def main():
    # Create results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Generate global seeds for all models
    global_seeds = generate_global_seeds()
    print("Global Seeds for this benchmark:", global_seeds)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results/hammurabi_benchmark_{timestamp}.jsonl"
    
    # Calculate total runs
    total_runs = len(MODELS) * RUNS_PER_MODEL
    current_run = 0
    
    print(f"Starting Hammurabi benchmark...")
    print(f"Models to test: {len(MODELS)}")
    print(f"Runs per model: {RUNS_PER_MODEL}")
    print(f"Total runs: {total_runs}")
    print(f"Results will be saved to: {output_file}")
    print()
    
    with open(output_file, 'w') as f:
        for model in MODELS:
            print(f"\nTesting model: {model}")
            for run in range(RUNS_PER_MODEL):
                current_run += 1
                seed = global_seeds[run]
                
                print(f"\nRun {run + 1}/{RUNS_PER_MODEL} (Overall progress: {current_run}/{total_runs})")
                
                # Run the game with the specific seed
                result = run_game(model, run + 1, seed)
                
                # Print error details if something went wrong
                if result.get("status") == "error":
                    print(f" Error in run {run + 1}: {result.get('error')}")
                    if "stderr" in result:
                        print("\nGame output:")
                        print("-" * 50)
                        print(result["stderr"])
                        print("-" * 50)
                else:
                    print(f" Run completed successfully! Score: {result.get('score', 'N/A')}")
                
                # Save result
                f.write(json.dumps(result) + '\n')
                f.flush()  # Ensure result is written immediately
                
                # Small delay between runs to avoid rate limiting
                time.sleep(1)
            
            print(f"\nCompleted all runs for {model}")
            
    print(f"\nBenchmark complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
