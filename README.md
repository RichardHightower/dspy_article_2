# DSPy Examples: Beyond Prompt Hacking

This project contains working examples from the chapter "Beyond Prompt Hacking: Why DSPy Is the Modern Approach to AI Programming".

## Overview

Learn how to move from fragile prompt engineering to robust, modular AI programming with DSPy. This example demonstrates:
- Basic Q&A modules
- Async summarization with structured outputs
- Multi-stage pipelines
- Document classification
- Optimization with feedback loops

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- API key for OpenAI or Anthropic (Claude) OR Ollama installed locally

## Setup

1. Clone this repository
2. Copy `.env.example` to `.env` and configure your LLM provider:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` to select your provider and model:
   - For OpenAI: Set `LLM_PROVIDER=openai` and add your API key
   - For Claude: Set `LLM_PROVIDER=anthropic` and add your API key
   - For Ollama: Set `LLM_PROVIDER=ollama` (install Ollama and pull phi3 model first)
4. Run the setup task:
   ```bash
   task setup
   ```

## Supported LLM Providers

### OpenAI
- Model: `gpt-4-turbo-preview`
- Requires: OpenAI API key

### Anthropic (Claude)
- Model: `claude-3-opus-20240229`
- Requires: Anthropic API key

### Ollama (Local)
- Model: `phi3:latest`
- Requires: Ollama installed and phi3 model pulled
- Install: `brew install ollama` (macOS) or see [ollama.ai](https://ollama.ai)
- Pull model: `ollama pull phi3`

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # LLM configuration
│   ├── main.py                # Entry point with all examples
│   ├── basic_examples.py      # Simple Q&A and summarization
│   ├── structured_outputs.py  # Async modules with Pydantic schemas
│   ├── pipelines.py           # Multi-stage processing
│   └── optimization.py        # Self-improving modules
├── tests/
│   └── test_modules.py        # Unit tests
├── .env.example               # Environment template
├── Taskfile.yml              # Task automation
└── pyproject.toml             # Poetry configuration
```

## Key Concepts Demonstrated

1. **Moving Beyond Prompts**: See how DSPy modules replace fragile prompt strings
2. **Modular Design**: Build reusable AI components with clear contracts
3. **Structured Outputs**: Use Pydantic schemas for reliable, validated responses
4. **Async Support**: Scale with non-blocking execution
5. **Self-Optimization**: Let DSPy improve your modules automatically

## Running Examples

Run all examples:
```bash
task run
```

Or run individual modules:
```bash
task run-basic           # Basic Q&A examples
task run-structured      # Structured outputs
task run-pipelines       # Multi-stage pipelines
task run-optimization    # Optimization examples
```

Direct Python execution:
```bash
poetry run python src/main.py
poetry run python src/basic_examples.py
poetry run python src/structured_outputs.py
poetry run python src/pipelines.py
poetry run python src/optimization.py
```

## Available Tasks

- `task setup` - Set up Python environment and install dependencies
- `task run` - Run the main example script
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task clean` - Clean up generated files

## Virtual Environment Setup Instructions

### Prerequisites
1. Install pyenv (if not already installed):
   ```bash
   # macOS
   brew install pyenv
   
   # Linux
   curl https://pyenv.run | bash
   ```

2. Add pyenv to your shell:
   ```bash
   # Add to ~/.zshrc or ~/.bashrc
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   
   # Reload shell
   source ~/.zshrc
   ```

### Setup Steps

1. **Install Python 3.12.8**:
   ```bash
   pyenv install 3.12.8
   ```

2. **Navigate to your project directory**:
   ```bash
   cd /path/to/dspy_chapter01
   ```

3. **Set local Python version**:
   ```bash
   pyenv local 3.12.8
   ```

4. **Install Poetry** (if not installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

5. **Install project dependencies**:
   ```bash
   poetry install
   ```

6. **Activate the virtual environment**:
   ```bash
   poetry config virtualenvs.in-project true
   source .venv/bin/activate
   ```

### Alternative: If you have Go Task installed
Simply run:
```bash
brew install go-task
task setup
```

### Configure your LLM provider

1. **Copy the example env file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit .env and set your provider**:
   ```bash
   # For OpenAI
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your-key-here
   OPENAI_MODEL=gpt-4-turbo-preview
   
   # For Anthropic/Claude
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=your-key-here
   ANTHROPIC_MODEL=claude-3-opus-20240229
   
   # For Ollama (local)
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=phi3:latest
   # Make sure Ollama is running: ollama serve
   # Pull the model: ollama pull phi3
   ```

### Verify setup
```bash
# Check Python version
python --version  # Should show 3.12.8

# Test imports
python -c "import dspy; print('DSPy installed successfully')"
```

### Run the example
```bash 
      poetry run python src/main.py
```


## Example Output

The examples demonstrate:
1. Basic Q&A with DSPy signatures
2. Code explanation with multiple outputs
3. Chain-of-thought reasoning for math problems
4. Complex pipelines for code analysis
5. Advanced features like optimization and async operations

## Troubleshooting

- **Ollama connection error**: Make sure Ollama is running (`ollama serve`)
- **API key errors**: Check your `.env` file has the correct keys
- **Model not found**: For Ollama, ensure you've pulled the model (`ollama pull phi3`)

## Learn More

- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [Full book: "Stop Wrestling with Prompts: How DSPy Transforms Fragile AI into Reliable Software"](https://rick-hightower.notion.site/DSPy-Book-20ad6bbdbbea80e1931befaa23292c5e)

