# Research Agent Game

A Docker-based research agent that can perform literature reviews, write papers, and manage code repositories.

## Prerequisites

- Docker
- Docker Compose
- OpenRouter API key

## Setup

1. Create a logs directory with proper permissions:
```bash
mkdir -p logs
chmod 777 logs  # Ensure Docker can write to this directory
```

2. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

## Running Multiple Instances

You can run multiple research agent instances simultaneously. Each instance will have its own:
- Logs directory
- State
- Model configuration

### Launch Commands

1. Start a new instance (in a new terminal):
```bash
# Basic instance with o3-mini model
GAME_INSTANCE_ID=exp1 docker-compose up

# Instance with different model
GAME_INSTANCE_ID=exp2 MODELS=anthropic/claude-3-opus docker-compose up

# Another instance
GAME_INSTANCE_ID=exp3 docker-compose up
```

2. View running instances:
```bash
docker ps
```

3. Stop a specific instance:
```bash
# Stop by instance ID
GAME_INSTANCE_ID=exp1 docker-compose down

# Or stop by container ID
docker stop <container_id>
```

### Instance Configuration

Each instance can be configured with environment variables:
- `GAME_INSTANCE_ID`: Unique identifier (e.g., exp1, exp2)
- `MODELS`: AI model to use (default: o3-mini)

### Logs and Output

- Each instance creates its own log directory: `logs/research_agent_<instance_id>`
- View logs in real-time in the terminal where you launched the instance
- Logs persist even after container shutdown

## Troubleshooting

1. Permission Issues:
```bash
# Fix logs directory permissions
chmod 777 logs

# If using SELinux
chcon -Rt container_file_t logs/
```

2. Container Issues:
```bash
# Remove all stopped containers
docker-compose down

# Rebuild image
docker-compose build --no-cache