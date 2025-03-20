import os
import subprocess
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dotenv import load_dotenv
import openai
from langchain.tools import Tool
from langchain.agents import AgentExecutor, BaseSingleActionAgent  # Import the base class
from langchain.schema import AgentAction, AgentFinish  # Import these classes
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field, BaseModel
import re
import time

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please add it to your .env file.")

# Initialize OpenAI client with DeepSeek base URL and API key from environment variables
openai.api_key = api_key
openai.api_base = "https://api.deepseek.com"

# Command history to track user's commands
command_history = []

# Dangerous command patterns to check
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"mkfs",
    r"dd\s+if=.*\s+of=/dev/",
    r"^\s*:(){ :\|: & };:",  # Fork bomb
    r"chmod\s+-R\s+777\s+/",
    r"> /dev/sda",
    r"wget.+\|\s*bash",
    r"curl.+\|\s*bash",
]

# Create a function to interface with the DeepSeek model with better error handling
def call_deepseek(prompt):
    try:
        print(f"Sending request to DeepSeek API at {openai.api_base}...")
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek API Error: {str(e)}")
        # Provide a fallback response so the agent can continue
        return "I'm having trouble connecting to my language processing service. Let me provide a simple response instead."

# Define tools for the agent
def execute_shell_command(command: str) -> str:
    """Execute a shell command and return the output."""
    # Check if command is potentially dangerous
    if is_dangerous_command(command):
        return "⚠️ This command seems potentially dangerous and has been blocked for safety reasons."
    
    # Add command to history
    command_history.append((time.strftime("%Y-%m-%d %H:%M:%S"), command))
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            text=True, 
            capture_output=True, 
            timeout=30
        )
        if result.returncode != 0:
            return f"Command executed with return code {result.returncode}.\nOutput: {result.stdout}\nError: {result.stderr}"
        return result.stdout if result.stdout else "Command executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"

def is_dangerous_command(command: str) -> bool:
    """Check if a command matches known dangerous patterns."""
    command = command.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return True
    return False

def get_command_explanation(command: str) -> str:
    """Get an explanation of what a command does."""
    response = call_deepseek(f"""
    Please provide a brief, clear explanation of what this shell command does:
    ```
    {command}
    ```
    Keep your explanation concise and focus on the effect of this command.
    """)
    return response

# Define tools
tools = [
    Tool(
        name="ShellExecutor",
        func=execute_shell_command,
        description="Executes shell commands on the system. Input should be a valid shell command."
    )
]

# Define the prompt template with enhanced system message
system_message = """You are an advanced shell command assistant that converts natural language requests into appropriate shell commands.

GUIDELINES:
1. When the user asks for something that can be done via shell commands, respond with the proper command.
2. Always consider the safest and most efficient approach to accomplish the task.
3. For complex tasks that require multiple commands, prefer using a single command with pipes or consider suggesting a small script.
4. For commands that might be destructive, include appropriate safeguards (like -i for rm).

EXAMPLES:
- If asked "show me the files in the current directory", respond with "ls -la"
- If asked "find all PDF files in my documents", respond with "find ~/Documents -name '*.pdf'"
- If asked "check system resource usage", respond with "top" or "htop" (if available)
- If asked "delete temporary files older than 7 days", respond with "find /tmp -type f -atime +7 -delete"

If you're unsure about a command or if the request isn't related to shell commands, explain that you're not sure.
Prioritize clarity and safety in your responses."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}")
])

# Create a simple LLM wrapper class to make DeepSeek compatible with LangChain
class DeepSeekWrapper:
    def invoke(self, prompt):
        if isinstance(prompt, list):
            # Convert the messages format to a single string for our simple wrapper
            prompt_text = "\n".join([f"{m.type}: {m.content}" for m in prompt])
        else:
            prompt_text = prompt
        return call_deepseek(prompt_text)
    
    def bind_functions(self, functions):
        # Implement this method to make the wrapper compatible with function calling
        # This is a minimal implementation that doesn't truly implement function calling
        return self

# Create a simple ReAct agent
def create_agent():
    # Define the ReAct logic
    def react_logic(input, agent_scratchpad=None):
        if agent_scratchpad is None:
            agent_scratchpad = []
        
        # Format thought process
        thoughts = ""
        for message in agent_scratchpad:
            thoughts += f"{message.type}: {message.content}\n"
        
        # Enhanced prompt for better shell command generation
        response = call_deepseek(f"""
You are an advanced shell command assistant that converts natural language requests into appropriate shell commands.

GUIDELINES:
1. When the user asks for something that can be done via shell commands, respond with the proper command.
2. Always consider the safest and most efficient approach to accomplish the task.
3. For complex tasks that require multiple commands, prefer using a single command with pipes or consider suggesting a small script.
4. For commands that might be destructive, include appropriate safeguards (like -i for rm).
5. For file operations, consider using relative paths over absolute when appropriate.

EXAMPLES:
- If asked "show me the files in the current directory", respond with "ls -la"
- If asked "find all PDF files in my documents", respond with "find ~/Documents -name '*.pdf'"
- If asked "check system resource usage", respond with "top" or "htop" (if available)
- If asked "delete temporary files older than 7 days", respond with "find /tmp -type f -atime +7 -delete"

Available tools:
ShellExecutor: Executes shell commands on the system. Input should be a valid shell command.

Previous thought process:
{thoughts}

User request: {input}

Think step by step about the appropriate shell command(s) to accomplish this task:
1. What is the user asking for?
2. Which command(s) would accomplish this task?
3. Is there a safer or more efficient way to do this?

Then respond in the format:
Thought: <your detailed reasoning>
Action: ShellExecutor
Action Input: <shell command to execute>

If you don't need a tool or want to provide a final answer:
Thought: <your reasoning>
Final Answer: <your final response to the user>
""")
        
        # Return the response
        return response

    return react_logic

# Update the SimpleAgent class to properly inherit from BaseSingleActionAgent
class SimpleAgent(BaseSingleActionAgent):
    """Agent that uses a logic function to determine actions."""
    
    # Define class fields
    logic_func: Callable = Field(exclude=True)  # Define logic_func as a field
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(self, logic_func: Callable, **kwargs):
        """Initialize the agent with a logic function."""
        super().__init__(logic_func=logic_func, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        """Define the expected input keys for the agent."""
        return ["input"]

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs) -> Union[AgentAction, AgentFinish]:
        # Extract previous steps
        agent_scratchpad = []
        for action, observation in intermediate_steps:
            agent_scratchpad.append(HumanMessage(content=f"Action: {action.tool}\nAction Input: {action.tool_input}"))
            agent_scratchpad.append(AIMessage(content=f"Observation: {observation}"))
        
        # Get input
        user_input = kwargs.get("input")
        
        # Call the logic function
        response_text = self.logic_func(user_input, agent_scratchpad)
        
        # Parse the response
        if "Action:" in response_text and "Action Input:" in response_text:
            # Extract action and action input
            action_match = response_text.split("Action:")[1].split("\n")[0].strip()
            action_input_match = response_text.split("Action Input:")[1].split("\n")[0].strip()
            
            # Return an AgentAction object
            return AgentAction(
                tool=action_match,
                tool_input=action_input_match,
                log=response_text
            )
        else:
            # Return an AgentFinish object
            return AgentFinish(
                return_values={"output": response_text},
                log=response_text
            )

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs) -> Union[AgentAction, AgentFinish]:
        """Asynchronous version of the plan method."""
        return self.plan(intermediate_steps, **kwargs)

# Create agent and agent executor
agent = SimpleAgent(create_agent())

# Create agent executor with the correct initialization
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # Changed from True to False to hide thinking steps
    handle_parsing_errors=True,
    max_iterations=5  # Add a limit to prevent infinite loops
)

def get_command_history():
    """Return formatted command history"""
    if not command_history:
        return "No commands have been executed yet."
    
    history = "Command History:\n"
    history += "--------------\n"
    for i, (timestamp, cmd) in enumerate(command_history[-10:], 1):
        history += f"{i}. [{timestamp}] {cmd}\n"
    return history

def main():
    print("🖥️  Enhanced Shell Command Agent (Type 'exit' to quit, 'history' to see command history, 'train' to improve agent)")
    print("-----------------------------------------------------------------------------")
    print(f"Using API base: {openai.api_base}")
    print(f"API Key is {'configured' if api_key else 'missing'}")
    
    # Verify API connection
    try:
        test_response = call_deepseek("Hello, this is a test message. Please respond with 'API connection successful'.")
        print(f"API Connection test: {'Successful' if 'success' in test_response.lower() else 'Failed with unexpected response'}")
    except Exception as e:
        print(f"API Connection test: Failed with error: {str(e)}")
    
    while True:
        user_input = input("\n>> ")  # Changed to simple arrow prompt
        
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'history':
            print(get_command_history())
            continue
        
        if user_input.lower() == 'train':
            print("Starting training agent...")
            try:
                # Import here to avoid circular import
                from training_agent import main as training_main
                training_main()
            except ImportError:
                print("❌ Training agent module not found. Please make sure training_agent.py is available.")
            except Exception as e:
                print(f"❌ Error starting training agent: {str(e)}")
            continue
        
        try:
            # Process the request through the agent
            response = agent_executor.invoke({"input": user_input})
            
            # Check if the response contains a shell command suggestion
            if "Final Answer: `" in response['output']:
                # Extract the command from the response
                command_match = re.search(r"Final Answer: `(.*?)`", response['output'])
                if command_match:
                    command = command_match.group(1).strip()
                    # Execute the extracted command directly
                    result = execute_shell_command(command)
                    print(result)
                else:
                    print(response['output'])
            else:
                # Regular response without command extraction
                print(response['output'])
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
