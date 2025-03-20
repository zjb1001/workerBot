# Excel Script Optimization Agent

This project provides an AI agent that automatically optimizes your Excel data processing scripts by iteratively comparing outputs with validation data and making improvements.

## How It Works

1. The agent takes your existing Python script that processes Excel data
2. It runs the script and compares its output with validation data
3. It identifies differences and suggests improvements
4. It tests the improved script and repeats until the output closely matches the validation data
5. It provides the optimized script as output

### Monitoring Agent

The project now includes a monitoring agent that:
- Tracks optimization progress and metrics
- Analyzes performance trends
- Provides guidance to improve optimization efficiency
- Generates visualization of the optimization journey
- Creates comprehensive reports on the optimization process

## Requirements

- Python 3.8+
- OpenAI or DeepSeek API key
- Your original Python script that processes Excel data
- Validation Excel file (the expected output)
- Input Excel file that the script should process

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/excel-script-optimizer.git
cd excel-script-optimizer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. **Important Note for DeepSeek API:** This project uses the official DeepSeek Python package. You'll need to:
   - Sign up for a DeepSeek API account at https://platform.deepseek.com/
   - Generate an API key from your account dashboard
   - Configure the API key using one of the methods described in the Configuration section below

## Configuration

There are three ways to configure the tool and provide your API keys:

### 1. Configuration File (Recommended)

Create a default configuration file:
```bash
python config.py --create-config
```

This creates a `config.json` file that you can edit to add your API keys and preferences:
```json
{
  "openai_api_key": "your-openai-api-key-here",
  "deepseek_api_key": "your-deepseek-api-key-here",
  "default_llm_provider": "openai",
  "default_model_name": "gpt-3.5-turbo",
  "max_iterations": 5,
  "output_directory": "./results"
}
```

### 2. Environment Variables

Copy the `.env.example` file to `.env` and edit it:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Or set environment variables directly:

```bash
# For OpenAI
export OPENAI_API_KEY=your-api-key-here

# For DeepSeek
export DEEPSEEK_API_KEY=your-api-key-here

# Other settings
export DEFAULT_LLM_PROVIDER=openai
export DEFAULT_MODEL_NAME=gpt-3.5-turbo
export MAX_ITERATIONS=5
```

### 3. Command Line Arguments

You can also provide configuration via command line arguments, which take precedence over config files and environment variables.

## Usage

### Standard Optimization

Run the optimization process:

```bash
python run_optimization.py \
  --script /path/to/your/excel_script.py \
  --validation /path/to/validation_data.xlsx \
  --input /path/to/input_data.xlsx
```

### Orchestrated Optimization with Monitoring

For advanced optimization with monitoring and guided improvement:

```bash
python orchestrator.py \
  --script /path/to/your/excel_script.py \
  --validation /path/to/validation_data.xlsx \
  --input /path/to/input_data.xlsx
```

The orchestrator combines the optimizer and monitor agents to:
1. Execute optimization iterations
2. Track performance metrics
3. Generate guidance for improving optimization
4. Adjust strategies based on monitoring insights
5. Produce detailed reports and visualizations

## Parameters

- `--script`: Path to your Excel processing Python script
- `--validation`: Path to the validation Excel file containing expected output
- `--input`: Path to the input Excel file that the script should process
- `--api-key`: (Optional) Override API key from config
- `--llm-provider`: (Optional) Which LLM provider to use - 'openai' or 'deepseek'
- `--model-name`: (Optional) Specific model name to use
- `--max-iterations`: (Optional) Maximum number of optimization iterations
- `--output-dir`: (Optional) Directory to save results
- `--config`: (Optional) Path to custom config file

### Orchestrator Parameters

- `--script`: Path to your Excel processing script
- `--validation`: Path to the validation Excel file containing expected output
- `--input`: Path to the input Excel file that the script should process
- `--api-key`: (Optional) Override API key from config
- `--llm-provider`: (Optional) Which LLM provider to use - 'openai' or 'deepseek'
- `--model-name`: (Optional) Specific model name to use
- `--max-iterations`: (Optional) Maximum number of optimization iterations
- `--output-dir`: (Optional) Directory to save results
- `--config`: (Optional) Path to custom config file
- `--no-guidance`: (Optional) Disable guidance from the monitoring agent

## Output

The optimization process will generate:

1. An optimized version of your script saved to the output directory
2. A JSON file with details about the optimization process and improvements made
3. Intermediate versions of the script created during optimization

With monitoring, additional outputs include:
1. A visualization of the optimization progress
2. Detailed logs of each iteration
3. AI-generated guidance reports
4. A comprehensive final report on the entire optimization process

## Example

Let's say you have a script `extract_data.py` that processes `input.xlsx` but doesn't properly match the expected output in `validation.xlsx`:

```bash
# Using configuration from config file or environment variables:
python run_optimization.py \
  --script ./extract_data.py \
  --validation ./validation.xlsx \
  --input ./input.xlsx

# Overriding the provider:
python run_optimization.py \
  --script ./extract_data.py \
  --validation ./validation.xlsx \
  --input ./input.xlsx \
  --llm-provider deepseek
```

After running, you'll get an optimized version of your script that produces output more closely matching the validation data.

# Shell Command Agent

This project implements an intelligent agent that can parse natural language instructions and convert them into shell commands. The agent uses the DeepSeek language model as its foundation and is built with the LangChain framework.

## Features

- Convert natural language requests to appropriate shell commands
- Execute shell commands and display their output
- Handle common tasks like printing directory trees

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your DeepSeek API key:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

## Usage

Run the agent:
