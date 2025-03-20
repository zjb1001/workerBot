import argparse
import os
from agentBot import AgentBot
import json
from config import config

def main():
    parser = argparse.ArgumentParser(description="Run the script optimizer agent")
    parser.add_argument("--script", required=True, help="Path to the Excel processing script")
    parser.add_argument("--validation", required=True, help="Path to validation Excel file")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--api-key", required=False, help="OpenAI or DeepSeek API key (override config)")
    parser.add_argument("--llm-provider", default=None, choices=["openai", "deepseek"],
                       help="The LLM provider to use (openai or deepseek)")
    parser.add_argument("--model-name", help="Specific model name to use")
    parser.add_argument("--max-iterations", type=int, default=None, help="Maximum optimization iterations")
    parser.add_argument("--output-dir", default=None, help="Directory to save results")
    parser.add_argument("--config", default=None, help="Path to custom config file")
    
    args = parser.parse_args()
    
    # If custom config file specified, create new Config instance
    if args.config:
        from config import Config
        custom_config = Config(args.config)
    else:
        custom_config = config  # Use default global config
    
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize optimizer - updated class name
        optimizer = AgentBot(
            args.script,
            args.validation,
            args.input,
            api_key,
            llm_provider,
            model_name
        )
        optimizer.max_iterations = max_iterations
        
        print(f"Starting optimization process with max {max_iterations} iterations")
        print(f"Using LLM provider: {llm_provider}")
        print(f"Original script: {args.script}")
        print(f"Validation data: {args.validation}")
        print(f"Input data: {args.input}")
        
        # Run optimization
        results = optimizer.optimize()
        
        # Save results
        script_name = os.path.splitext(os.path.basename(args.script))[0]
        result_path = os.path.join(output_dir, f"{script_name}_optimization_results.json")
        final_script_path = os.path.join(output_dir, f"{script_name}_optimized.py")
        
        # Save the final optimized script
        with open(final_script_path, 'w') as f:
            f.write(results["final_script"])
        
        # Save detailed results
        with open(result_path, 'w') as f:
            # Don't save full scripts in the results to avoid duplication
            summary_results = {
                "initial_script_path": args.script,
                "final_script_path": final_script_path,
                "llm_provider": llm_provider,
                "iterations": len(results["improvement_history"]),
                "improvement_summary": [
                    {
                        "iteration": item["iteration"],
                        "diff_summary": "\n".join(item["diff"].split("\n")[:10]) + "..."
                    }
                    for item in results["improvement_history"]
                ]
            }
            json.dump(summary_results, f, indent=2)
        
        print(f"\nOptimization complete!")
        print(f"Final optimized script saved to: {final_script_path}")
        print(f"Detailed results saved to: {result_path}")
    
    except ImportError as e:
        print(f"\nError: {str(e)}")
        if llm_provider == "deepseek":
            print("\nDeepSeek API installation issue detected. Please see the README.md file for")
            print("instructions on how to install the DeepSeek client package correctly.")
        else:
            print("Please make sure all required packages are installed correctly.")

if __name__ == "__main__":
    main()
