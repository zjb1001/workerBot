import os
import json
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from agentBot import DeepSeekLLM  # Updated import

class OptimizationMonitor:
    """Monitoring agent that tracks the optimization process and provides guidance."""
    
    def __init__(self, api_key: str, llm_provider: str = "openai", model_name: Optional[str] = None):
        """Initialize the monitoring agent.
        
        Args:
            api_key: API key for the LLM provider
            llm_provider: 'openai' or 'deepseek'
            model_name: Specific model name to use
        """
        self.api_key = api_key
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.observations = []
        self.metrics = []
        self.guidance_history = []
        self.log_path = os.path.join("results", "monitoring")
        self.target_score = 95
        
        # Initialize LLM for guidance generation
        if llm_provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = OpenAI(temperature=0.2, model_name=model_name or "gpt-3.5-turbo")
        elif llm_provider == "deepseek":
            self.llm = DeepSeekLLM(
                api_key=api_key,
                model_name=model_name or "deepseek-chat",
                temperature=0.2
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
            
        # Create log directory
        os.makedirs(self.log_path, exist_ok=True)
        
    def record_iteration(self, 
                        iteration: int, 
                        prompt: str, 
                        script_changes: Dict[str, Any], 
                        similarity_score: float, 
                        differences: Dict[str, Any],
                        execution_time: float):
        """Record information about an optimization iteration.
        
        Args:
            iteration: The iteration number
            prompt: The prompt used for this iteration
            script_changes: Changes made to the script (diff)
            similarity_score: Current similarity score
            differences: Differences between output and validation data
            execution_time: Time taken for this iteration
        """
        timestamp = datetime.now().isoformat()
        
        observation = {
            "timestamp": timestamp,
            "iteration": iteration,
            "similarity_score": similarity_score,
            "execution_time": execution_time,
            "differences": differences,
            "prompt": prompt,
            "script_changes": script_changes,
        }
        
        self.observations.append(observation)
        
        # Extract metrics
        self.metrics.append({
            "iteration": iteration,
            "timestamp": timestamp,
            "similarity_score": similarity_score,
            "execution_time": execution_time,
            "num_differences": len(differences) if isinstance(differences, dict) else 0,
        })
        
        # Save observation
        self._save_observation(observation)
        
    def _save_observation(self, observation: Dict[str, Any]):
        """Save observation to disk."""
        filename = f"observation_iter_{observation['iteration']}.json"
        filepath = os.path.join(self.log_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(observation, f, indent=2)
    
    def analyze_progress(self) -> Dict[str, Any]:
        """Analyze current optimization progress and trends.
        
        Returns:
            Dict containing analysis results
        """
        if not self.metrics:
            return {"status": "No data available"}
            
        df = pd.DataFrame(self.metrics)
        
        # Calculate improvement rate
        if len(df) >= 2:
            first_score = df.iloc[0]["similarity_score"]
            current_score = df.iloc[-1]["similarity_score"]
            total_iterations = len(df)
            improvement_rate = (current_score - first_score) / total_iterations
        else:
            improvement_rate = 0
            
        # Check if progress is stalling
        is_stalling = False
        stall_threshold = 1.0  # Minimal improvement needed
        if len(df) >= 3:
            recent_scores = df.tail(3)["similarity_score"].tolist()
            if max(recent_scores) - min(recent_scores) < stall_threshold:
                is_stalling = True
                
        # Estimate iterations to reach target
        iterations_to_target = 0
        if improvement_rate > 0:
            current_score = df.iloc[-1]["similarity_score"]
            remaining_improvement = self.target_score - current_score
            iterations_to_target = int(remaining_improvement / improvement_rate) + 1
            
        return {
            "current_score": df.iloc[-1]["similarity_score"] if not df.empty else 0,
            "total_iterations": len(df),
            "improvement_rate": improvement_rate,
            "is_stalling": is_stalling,
            "iterations_to_target": iterations_to_target,
            "execution_times": df["execution_time"].tolist() if not df.empty else [],
            "scores": df["similarity_score"].tolist() if not df.empty else [],
        }
    
    def generate_guidance(self) -> Dict[str, str]:
        """Generate guidance for improving the optimization process.
        
        Returns:
            Dict containing guidance and reasoning
        """
        if not self.observations or len(self.observations) < 1:
            return {
                "guidance": "Continue with initial optimization.",
                "reasoning": "Not enough data to provide guidance yet."
            }
        
        analysis = self.analyze_progress()
        
        # Get the most recent observation
        latest_obs = self.observations[-1]
        
        # Create an analysis summary for the prompt
        if len(self.observations) >= 2:
            prev_obs = self.observations[-2]
            score_change = latest_obs["similarity_score"] - prev_obs["similarity_score"]
            score_trend = "improved" if score_change > 0 else "declined" if score_change < 0 else "unchanged"
        else:
            score_trend = "baseline"
        
        # Construct the prompt for the LLM
        prompt = f"""As an AI optimization coach, analyze the current Excel script optimization process and suggest improvements.

Current state:
- Iteration: {latest_obs["iteration"]}
- Current similarity score: {latest_obs["similarity_score"]:.2f}/100 (score has {score_trend} from previous iteration)
- Target score: {self.target_score}/100
- Optimization progress is {"stalling" if analysis["is_stalling"] else "progressing"}

Recent differences between the output and validation data:
{json.dumps(latest_obs["differences"], indent=2)}

Based on this information, provide specific guidance on:
1. What aspects of the script should be prioritized for improvement?
2. What approach should be taken to address the current differences?
3. Should the optimization strategy change, and if so, how?

Format your response as:
{{
  "guidance": "Concise guidance statement",
  "reasoning": "Detailed explanation of your reasoning",
  "suggested_focus": "Specific aspect to focus on",
  "prompt_addition": "Text to add to the next optimization prompt"
}}
"""
        
        # Get guidance from LLM
        response = self.llm(prompt)
        
        # Parse response
        try:
            guidance = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if not valid JSON
            guidance = {
                "guidance": "Continue optimization focusing on the biggest differences.",
                "reasoning": "Could not parse structured guidance. Focus on main differences.",
                "suggested_focus": "Major differences",
                "prompt_addition": "Pay special attention to columns with the largest discrepancies."
            }
        
        # Add iteration info
        guidance["iteration"] = latest_obs["iteration"]
        guidance["timestamp"] = datetime.now().isoformat()
        
        # Save to history
        self.guidance_history.append(guidance)
        self._save_guidance(guidance)
        
        return guidance
    
    def _save_guidance(self, guidance: Dict[str, Any]):
        """Save guidance to disk."""
        filename = f"guidance_iter_{guidance['iteration']}.json"
        filepath = os.path.join(self.log_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(guidance, f, indent=2)
    
    def visualize_progress(self, save_path: Optional[str] = None):
        """Generate visualizations of the optimization progress.
        
        Args:
            save_path: Path to save the visualization, if None, just displays the plot
        """
        if not self.metrics:
            print("No data available to visualize")
            return
            
        df = pd.DataFrame(self.metrics)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot similarity score
        ax1.plot(df["iteration"], df["similarity_score"], 'b-o', linewidth=2)
        ax1.axhline(y=self.target_score, color='r', linestyle='--', label=f'Target ({self.target_score}%)')
        ax1.set_title('Optimization Progress: Similarity Score')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Similarity Score (%)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot execution time
        ax2.plot(df["iteration"], df["execution_time"], 'g-o', linewidth=2)
        ax2.set_title('Execution Time per Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the optimization process.
        
        Returns:
            Dict containing the report data
        """
        analysis = self.analyze_progress()
        
        if not self.observations:
            return {"status": "No optimization data available"}
        
        # Prepare data for report
        report = {
            "start_time": self.observations[0]["timestamp"],
            "end_time": self.observations[-1]["timestamp"],
            "total_iterations": len(self.observations),
            "final_score": self.observations[-1]["similarity_score"],
            "target_reached": self.observations[-1]["similarity_score"] >= self.target_score,
            "improvement_rate": analysis["improvement_rate"],
            "execution_time_trend": "increasing" if len(analysis["execution_times"]) > 1 and analysis["execution_times"][-1] > analysis["execution_times"][0] else "decreasing",
            "total_execution_time": sum(obs["execution_time"] for obs in self.observations),
            "guidance_summary": [
                {
                    "iteration": g["iteration"],
                    "guidance": g["guidance"],
                    "suggested_focus": g.get("suggested_focus", "Not specified")
                }
                for g in self.guidance_history
            ],
            "optimization_journey": [
                {
                    "iteration": obs["iteration"],
                    "score": obs["similarity_score"],
                    "time": obs["execution_time"]
                }
                for obs in self.observations
            ]
        }
        
        # Generate visualization
        vis_path = os.path.join(self.log_path, "optimization_progress.png")
        self.visualize_progress(save_path=vis_path)
        report["visualization_path"] = vis_path
        
        # Save report
        report_path = os.path.join(self.log_path, "optimization_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report

if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the monitoring agent")
    parser.add_argument("--api-key", required=True, help="API key for the LLM")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "deepseek"], 
                      help="LLM provider to use")
    
    args = parser.parse_args()
    
    monitor = OptimizationMonitor(
        api_key=args.api_key,
        llm_provider=args.llm_provider
    )
    
    # Simulate some observations
    for i in range(1, 4):
        monitor.record_iteration(
            iteration=i,
            prompt=f"Test prompt for iteration {i}",
            script_changes={"diff": f"Changes in iteration {i}"},
            similarity_score=50 + i*10,  # Simulated improvement
            differences={"column1": "5 mismatches"},
            execution_time=10.5 + i
        )
        
        # Generate guidance
        guidance = monitor.generate_guidance()
        print(f"\n--- Guidance for iteration {i} ---")
        print(f"Guidance: {guidance['guidance']}")
        print(f"Reasoning: {guidance['reasoning']}")
        print(f"Suggested focus: {guidance.get('suggested_focus', 'None')}")
    
    # Generate report
    report = monitor.generate_report()
    print("\n--- Final Report Summary ---")
    print(f"Total iterations: {report['total_iterations']}")
    print(f"Final score: {report['final_score']}")
    print(f"Target reached: {report['target_reached']}")
    print(f"Report saved to: {os.path.join(monitor.log_path, 'optimization_report.json')}")
