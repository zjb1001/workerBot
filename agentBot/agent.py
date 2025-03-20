import os
import pandas as pd
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain
import subprocess
import json
import difflib
from typing import Optional, Dict, Any

# Add import for DeepSeek API support
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

class DeepSeekLLM(BaseLLM):
    """Custom LLM wrapper for DeepSeek API."""
    
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", **kwargs):
        """Initialize the DeepSeek LLM wrapper."""
        self.client = None  # Initialize here instead of at class level
        self.model_name = model_name
        self.model_kwargs = kwargs or {}
        
        try:
            # Use the official deepseek package from PyPI
            import deepseek
            # Configure API key
            deepseek.api_key = api_key
            # Save client for later use
            self.client = deepseek
            print("Using official deepseek package from PyPI")
        except ImportError:
            raise ImportError(
                "Could not import DeepSeek API package. "
                "Please install it with `pip install deepseek`."
            )
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the DeepSeek API."""
        # Merge keyword arguments
        merged_kwargs = {**self.model_kwargs, **kwargs}
        if stop:
            merged_kwargs["stop"] = stop
        
        try:
            # Use the official deepseek API format
            completion = self.client.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **merged_kwargs
            )
            
            # Extract the output text from the response
            return completion.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Failed to generate with DeepSeek API: {str(e)}")


class ScriptOptimizer:
    def __init__(self, target_script_path, validation_data_path, input_data_path, api_key, llm_provider="openai", model_name=None):
        # Initialize with paths to the script and validation data
        self.target_script_path = target_script_path
        self.validation_data_path = validation_data_path
        self.input_data_path = input_data_path
        self.current_script_path = target_script_path
        self.iteration = 0
        self.max_iterations = 10
        self.improvement_history = []
        self.llm_provider = llm_provider.lower()
        
        # Set up API keys
        if self.llm_provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = OpenAI(temperature=0, model_name=model_name or "gpt-3.5-turbo")
        elif self.llm_provider == "deepseek":
            # Initialize DeepSeek LLM
            self.llm = DeepSeekLLM(
                api_key=api_key,
                model_name=model_name or "deepseek-chat",
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Choose 'openai' or 'deepseek'.")
        
        # Define tools for the agent
        self.tools = [
            Tool(
                name="run_script",
                func=self.run_script,
                description="Run the current version of the script and get the output"
            ),
            Tool(
                name="compare_outputs",
                func=self.compare_with_validation,
                description="Compare the script's output with the validation data"
            ),
            Tool(
                name="improve_script",
                func=self.improve_script,
                description="Generate an improved version of the script based on comparison results"
            ),
            Tool(
                name="save_improved_script",
                func=self.save_improved_script,
                description="Save the improved script and make it the current version"
            )
        ]
        
        # Create the agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the LangChain agent with the tools"""
        prefix = """You are an AI agent tasked with improving a Python script that processes Excel data.
        The script extracts information from an input Excel file and saves it to an output Excel file.
        Your goal is to modify the script so its output matches the validation data as closely as possible.
        You have access to the following tools:"""
        
        suffix = """Begin by understanding what the script currently does,
        then iteratively improve it until its output matches the validation data.
        
        Let's work through this step by step.
        Thought: I need to first examine the current script and its output.
        """
        
        prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "agent_scratchpad"]
        )
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        self.agent = AgentExecutor.from_agent_and_tools(
            agent=ZeroShotAgent(llm_chain=llm_chain, tools=self.tools),
            tools=self.tools,
            verbose=True
        )
    
    def run_script(self, args=None):
        """Run the current version of the script and return the results"""
        try:
            result = subprocess.run(
                ["python", self.current_script_path, self.input_data_path],
                capture_output=True,
                text=True,
                check=True
            )
            # Assume the script creates an output file with a predictable name
            output_path = self.input_data_path.replace('.xlsx', '_output.xlsx')
            if os.path.exists(output_path):
                # Read and return a sample of the output data
                output_data = pd.read_excel(output_path)
                return f"Script executed successfully. Output sample (first 5 rows):\n{output_data.head().to_string()}"
            else:
                return f"Script executed but no output file found at {output_path}"
        except subprocess.CalledProcessError as e:
            return f"Error running script: {e.stderr}"
    
    def compare_with_validation(self, args=None):
        """Compare the script's output with validation data"""
        try:
            # Assume the script creates an output file with a predictable name
            output_path = self.input_data_path.replace('.xlsx', '_output.xlsx')
            
            if not os.path.exists(output_path):
                return "No output file found to compare"
            
            output_data = pd.read_excel(output_path)
            validation_data = pd.read_excel(self.validation_data_path)
            
            # Calculate differences and similarities
            differences = {}
            
            # Compare shapes
            if output_data.shape != validation_data.shape:
                differences["shape"] = f"Output has shape {output_data.shape}, but validation has {validation_data.shape}"
            
            # Compare columns
            missing_cols = set(validation_data.columns) - set(output_data.columns)
            extra_cols = set(output_data.columns) - set(validation_data.columns)
            if missing_cols:
                differences["missing_columns"] = f"Missing columns: {missing_cols}"
            if extra_cols:
                differences["extra_columns"] = f"Extra columns: {extra_cols}"
            
            # For common columns, compare values (sample of differences)
            common_cols = set(validation_data.columns) & set(output_data.columns)
            value_diffs = {}
            for col in common_cols:
                # Check for numerical columns and calculate mean difference
                if pd.api.types.is_numeric_dtype(validation_data[col]) and pd.api.types.is_numeric_dtype(output_data[col]):
                    # Calculate mean absolute difference
                    if len(output_data) == len(validation_data):
                        diff = (validation_data[col] - output_data[col]).abs().mean()
                        if diff > 0.001:  # Threshold for reporting difference
                            value_diffs[col] = f"Mean absolute difference: {diff}"
                # For non-numeric or if lengths differ, show sample differences
                else:
                    # Get sample of differing rows
                    if len(output_data) == len(validation_data):
                        mismatches = (output_data[col] != validation_data[col]).sum()
                        if mismatches > 0:
                            value_diffs[col] = f"{mismatches} mismatches out of {len(validation_data)}"
            
            if value_diffs:
                differences["value_differences"] = value_diffs
            
            # Calculate an overall similarity score
            similarity_score = 0
            max_score = 100
            if not differences:
                similarity_score = max_score
            elif "shape" in differences:
                # Major structural difference
                similarity_score = 30
            else:
                # Start with base score and deduct for differences
                similarity_score = 80
                if "missing_columns" in differences:
                    similarity_score -= 10 * len(missing_cols) / len(validation_data.columns)
                if "extra_columns" in differences:
                    similarity_score -= 5 * len(extra_cols) / len(validation_data.columns)
                if "value_differences" in differences:
                    similarity_score -= 30 * len(value_diffs) / len(common_cols)
            
            result = {
                "similarity_score": similarity_score,
                "differences": differences if differences else "No significant differences found",
                "conclusion": "Perfect match!" if similarity_score >= max_score else "Needs improvement"
            }
            
            return json.dumps(result, indent=2)
        
        except Exception as e:
            return f"Error comparing data: {str(e)}"
    
    def improve_script(self, comparison_result):
        """Generate an improved version of the script based on comparison results"""
        try:
            # Read the current script
            with open(self.current_script_path, 'r') as file:
                current_script = file.read()
            
            # Create a prompt for the LLM
            prompt = f"""I need you to improve the following Python script that processes Excel data.
            
            The current script:
            ```python
            {current_script}
            ```
            
            The comparison with validation data shows:
            {comparison_result}
            
            Please create an improved version of this script that addresses the differences 
            and makes the output match the validation data more closely.
            
            Return ONLY the improved Python code without explanations or markdown formatting.
            """
            
            # Get improved script from LLM
            improved_script = self.llm(prompt)
            
            # Clean up the response to extract just the code
            if "```python" in improved_script:
                improved_script = improved_script.split("```python")[1].split("```")[0].strip()
            elif "```" in improved_script:
                improved_script = improved_script.split("```")[1].split("```")[0].strip()
            
            # Store the changes for review
            diff = list(difflib.unified_diff(
                current_script.splitlines(),
                improved_script.splitlines(),
                lineterm='',
                n=3
            ))
            
            changes = {
                "iteration": self.iteration + 1,
                "diff": '\n'.join(diff),
                "improved_script": improved_script
            }
            
            self.improvement_history.append(changes)
            
            return improved_script
        
        except Exception as e:
            return f"Error improving script: {str(e)}"
    
    def save_improved_script(self, improved_script):
        """Save the improved script and make it the current version"""
        try:
            # Create a new file for this iteration
            self.iteration += 1
            script_name = os.path.basename(self.target_script_path)
            script_base, script_ext = os.path.splitext(script_name)
            new_script_path = os.path.join(
                os.path.dirname(self.target_script_path),
                f"{script_base}_v{self.iteration}{script_ext}"
            )
            
            # Save the improved script
            with open(new_script_path, 'w') as file:
                file.write(improved_script)
            
            # Update the current script path
            self.current_script_path = new_script_path
            
            return f"Saved improved script as {new_script_path}"
        
        except Exception as e:
            return f"Error saving script: {str(e)}"
    
    def optimize(self):
        """Run the optimization process"""
        print(f"Starting script optimization. Max iterations: {self.max_iterations}")
        
        # Read the initial script for reference
        with open(self.target_script_path, 'r') as file:
            initial_script = file.read()
        
        iteration_input = f"I need to optimize this Excel processing script: {self.target_script_path}"
        
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")
            result = self.agent.run(iteration_input)
            print(result)
            
            # Check comparison results from the latest improvement
            if self.improvement_history:
                latest_comparison = json.loads(self.compare_with_validation())
                similarity_score = latest_comparison.get("similarity_score", 0)
                
                print(f"Current similarity score: {similarity_score}/100")
                
                # If we've reached a high enough similarity, we're done
                if similarity_score >= 95:
                    print("Optimization successful! Reached 95% or higher similarity.")
                    break
            
            # Update the input for the next iteration
            iteration_input = f"Continue optimizing the script. Current version: {self.current_script_path}"
        
        # Return the final optimized script
        with open(self.current_script_path, 'r') as file:
            final_script = file.read()
        
        print("\nOptimization complete!")
        print(f"Initial script: {self.target_script_path}")
        print(f"Final script: {self.current_script_path}")
        
        return {
            "initial_script": initial_script,
            "final_script": final_script,
            "improvement_history": self.improvement_history
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize an Excel processing script")
    parser.add_argument("script_path", help="Path to the script to optimize")
    parser.add_argument("validation_data", help="Path to the validation data Excel file")
    parser.add_argument("input_data", help="Path to the input data Excel file")
    parser.add_argument("--api-key", required=True, help="OpenAI or DeepSeek API key")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "deepseek"], 
                       help="The LLM provider to use (openai or deepseek)")
    parser.add_argument("--model-name", help="Specific model name to use")
    
    args = parser.parse_args()
    
    optimizer = ScriptOptimizer(
        args.script_path,
        args.validation_data,
        args.input_data,
        args.api_key,
        args.llm_provider,
        args.model_name
    )
    
    result = optimizer.optimize()
