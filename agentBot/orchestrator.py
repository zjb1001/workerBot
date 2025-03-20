import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from agentBot import AgentBot  # Updated import
from monitor_agent import OptimizationMonitor
from config import config

class OptimizationOrchestrator:
    """Orchestrates the script optimization process with monitoring."""
    
    def __init__(
        self, 
        script_path: str,
        validation_data: str,
        input_data: str,
        api_key: str,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        max_iterations: int = 10,
        output_dir: str = "./results",
        apply_guidance: bool = True
    ):
        """Initialize the orchestrator.
        
        Args:
            script_path: Path to the script to optimize
            validation_data: Path to validation Excel file
            input_data: Path to input Excel file
            api_key: API key for the LLM
            llm_provider: LLM provider (openai or deepseek)
            model_name: Model name to use
            max_iterations: Maximum number of optimization iterations
            output_dir: Directory to save results
            apply_guidance: Whether to apply guidance from the monitor
        """
        self.script_path = script_path
        self.validation_data = validation_data
        self.input_data = input_data
        self.api_key = api_key
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.apply_guidance = apply_guidance
        
        # Initialize optimizer and monitor - updated class name
        self.optimizer = AgentBot(
            script_path,
            validation_data,
            input_data,
            api_key,
            llm_provider,
            model_name
        )
        self.optimizer.max_iterations = max_iterations
        
        self.monitor = OptimizationMonitor(
            api_key=api_key,
            llm_provider=llm_provider,
            model_name=model_name
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def run_optimization(self):
        """Run the optimization process with monitoring."""
        print(f"Starting orchestrated optimization with max {self.max_iterations} iterations")
        print(f"Using LLM provider: {self.llm_provider}")
        print(f"Script: {self.script_path}")
        print(f"Validation data: {self.validation_data}")
        print(f"Input data: {self.input_data}")
        print(f"Monitoring: {'Enabled with guidance' if self.apply_guidance else 'Enabled (no guidance)'}")
        
        # Read the initial script
        with open(self.script_path, 'r') as file:
            initial_script = file.read()
        
        current_script_path = self.script_path
        
        # Initial prompt
        base_prompt = f"I need to optimize this Excel processing script: {self.script_path}"
        current_prompt = base_prompt
        
        # Main optimization loop
        for i in range(self.max_iterations):
            iteration_start_time = time.time()
            iteration = i + 1
            
            print(f"\n=== Iteration {iteration}/{self.max_iterations} ===")
            print(f"Current script: {current_script_path}")
            
            # If guidance is enabled and we have at least one iteration
            guidance_addition = ""
            if self.apply_guidance and i > 0:
                # Get guidance from monitor
                guidance = self.monitor.generate_guidance()
                print("\n--- Monitor Guidance ---")
                print(f"Guidance: {guidance['guidance']}")
                print(f"Focus on: {guidance.get('suggested_focus', 'Not specified')}")
                
                # Apply guidance to prompt
                if 'prompt_addition' in guidance:
                    guidance_addition = guidance['prompt_addition']
                    current_prompt = f"{base_prompt}\n\nAdditional guidance: {guidance_addition}"
            
            # Run the current iteration
            # Note: We directly interact with optimizer internals for more control
            self.optimizer.current_script_path = current_script_path
            self.optimizer.iteration = i
            
            # Execute agent with current prompt
            print("\n--- Running Optimizer Agent ---")
            result = self.optimizer.agent.run(current_prompt)
            print(result)
            
            # Get comparison results
            latest_comparison_json = self.optimizer.compare_with_validation()
            try:
                latest_comparison = json.loads(latest_comparison_json)
                similarity_score = latest_comparison.get("similarity_score", 0)
                differences = latest_comparison.get("differences", {})
            except json.JSONDecodeError:
                similarity_score = 0
                differences = {"error": "Failed to parse comparison results"}
                
            print(f"Current similarity score: {similarity_score}/100")
            
            # Extract script changes if available
            script_changes = {}
            if self.optimizer.improvement_history:
                latest_improvement = self.optimizer.improvement_history[-1]
                script_changes = {
                    "diff": latest_improvement["diff"],
                    "iteration": latest_improvement["iteration"]
                }
            
            # Calculate iteration time
            iteration_time = time.time() - iteration_start_time
            
            # Record data in monitor
            self.monitor.record_iteration(
                iteration=iteration,
                prompt=current_prompt,
                script_changes=script_changes,
                similarity_score=similarity_score,
                differences=differences,
                execution_time=iteration_time
            )
            
            # Update the current script path for next iteration
            if self.optimizer.current_script_path != current_script_path:
                current_script_path = self.optimizer.current_script_path
                
            # Check if we've reached the target similarity
            if similarity_score >= 95:
                print("\nOptimization target reached! Similarity score >= 95%")
                break
                
            # Update the base prompt for next iteration
            base_prompt = f"Continue optimizing the script. Current version: {current_script_path}"
            current_prompt = base_prompt
            
        # End of optimization
        
        # Get the final script
        with open(current_script_path, 'r') as file:
            final_script = file.read()
            
        # Generate the final report
        print("\nGenerating final report...")
        report = self.monitor.generate_report()
        
        # Save the final optimized script
        script_name = os.path.splitext(os.path.basename(self.script_path))[0]
        final_script_path = os.path.join(self.output_dir, f"{script_name}_optimized.py")
        
        with open(final_script_path, 'w') as f:
            f.write(final_script)
            
        print("\nOrchestrated optimization complete!")
        print(f"Final script saved to: {final_script_path}")
        print(f"Monitoring report: {os.path.join(self.monitor.log_path, 'optimization_report.json')}")
        print(f"Visualization: {os.path.join(self.monitor.log_path, 'optimization_progress.png')}")
        
        return {
            "initial_script": initial_script,
            "final_script": final_script,
            "report": report,
            "improvement_history": self.optimizer.improvement_history
        }

def main():
    parser = argparse.ArgumentParser(description="Run orchestrated script optimization with monitoring")
    parser.add_argument("--script", required=True, help="Path to the Excel processing script")
    parser.add_argument("--validation", required=True, help="Path to validation Excel file")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--api-key", required=False, help="API key (override config)")
    parser.add_argument("--llm-provider", default=None, help="LLM provider to use (openai or deepseek)")
    parser.add_argument("--model-name", help="Specific model name to use")
    parser.add_argument("--max-iterations", type=int, default=None, help="Maximum optimization iterations")
    parser.add_argument("--output-dir", default=None, help="Directory to save results")
    parser.add_argument("--config", default=None, help="Path to custom config file")
    parser.add_argument("--no-guidance", action="store_true", help="Disable guidance from monitor")
    
    args = parser.parse_args()
    
    # If custom config file specified, create new Config instance
    if args.config:
        from config import Config
        custom_config = Config(args.config)
    else:
        custom_config = config
    
    # Get provider from args or config
    llm_provider = args.llm_provider or custom_config.get("default_llm_provider", "openai")
    
    # Get API key from args, or from config based on provider
    api_key = args.api_key or custom_config.get_api_key(llm_provider)
    
    if not api_key:
        raise ValueError(
            f"No API key found for {llm_provider}. Please provide it via --api-key, "
            f"environment variable {llm_provider.upper()}_API_KEY, or in config file."
        )
    
    # Get other parameters from args or config
    model_name = args.model_name or custom_config.get("default_model_name")
    max_iterations = args.max_iterations or custom_config.get("max_iterations", 5)
    output_dir = args.output_dir or custom_config.get("output_directory", "./results")
    
    # Create orchestrator
    orchestrator = OptimizationOrchestrator(
        script_path=args.script,
        validation_data=args.validation,
        input_data=args.input,
        api_key=api_key,
        llm_provider=llm_provider,
        model_name=model_name,
        max_iterations=max_iterations,
        output_dir=output_dir,
        apply_guidance=not args.no_guidance
    )
    
    # Run optimization
    orchestrator.run_optimization()

if __name__ == "__main__":
    main()
