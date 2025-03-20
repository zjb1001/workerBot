import os
import json
import time
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

# Import components from shell_agent.py
from shell_agent import call_deepseek, execute_shell_command, is_dangerous_command

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "command_feedback.json")
METRICS_FILE = os.path.join(DATA_DIR, "performance_metrics.json")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class TrainingDataCollector:
    """Collects and manages training data for the shell agent."""
    
    def __init__(self):
        self.feedback_data = self._load_feedback_data()
        self.session_data = []
    
    def _load_feedback_data(self) -> List[Dict[str, Any]]:
        """Load existing feedback data or initialize if not exists."""
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Feedback file corrupted, creating new one")
                return []
        return []
    
    def save_interaction(self, user_request: str, generated_command: str, 
                         execution_result: str, duration: float) -> None:
        """Save an interaction to the session data."""
        self.session_data.append({
            "timestamp": datetime.now().isoformat(),
            "user_request": user_request,
            "generated_command": generated_command,
            "execution_result": execution_result,
            "duration": duration,
            "feedback": None  # To be filled later by user
        })
    
    def add_user_feedback(self, interaction_index: int, 
                          rating: int, correct_command: Optional[str] = None,
                          comments: Optional[str] = None) -> None:
        """Add user feedback to a specific interaction."""
        if 0 <= interaction_index < len(self.session_data):
            self.session_data[interaction_index]["feedback"] = {
                "rating": rating,  # 1-5 scale
                "correct_command": correct_command,
                "comments": comments
            }
    
    def save_feedback_data(self) -> None:
        """Save session data to persistent storage."""
        # Add session data to overall feedback data
        self.feedback_data.extend([x for x in self.session_data if x["feedback"] is not None])
        
        # Save to file
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
        
        print(f"Saved {len(self.session_data)} new interactions with feedback")

class PerformanceEvaluator:
    """Evaluates shell agent performance based on collected data."""
    
    def __init__(self, data_collector: TrainingDataCollector):
        self.data_collector = data_collector
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, List]:
        """Load existing metrics or initialize if not exists."""
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._initialize_metrics()
        return self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict[str, List]:
        """Initialize the metrics dictionary."""
        return {
            "timestamps": [],
            "avg_rating": [],
            "command_accuracy": [],
            "response_time": [],
            "success_rate": []
        }
    
    def calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate metrics based on recent feedback data."""
        # Only use data with feedback
        recent_data = [x for x in self.data_collector.session_data if x["feedback"] is not None]
        
        if not recent_data:
            return {
                "avg_rating": 0,
                "command_accuracy": 0,
                "response_time": 0,
                "success_rate": 0
            }
        
        # Calculate metrics
        avg_rating = np.mean([x["feedback"]["rating"] for x in recent_data])
        response_time = np.mean([x["duration"] for x in recent_data])
        
        # Determine success rate (rating >= 4 is considered success)
        success_count = sum(1 for x in recent_data if x["feedback"]["rating"] >= 4)
        success_rate = success_count / len(recent_data) if recent_data else 0
        
        # Command accuracy (if correct_command is None, assume command was correct)
        correct_count = sum(1 for x in recent_data if x["feedback"]["correct_command"] is None)
        command_accuracy = correct_count / len(recent_data) if recent_data else 0
        
        return {
            "avg_rating": float(avg_rating),
            "command_accuracy": float(command_accuracy),
            "response_time": float(response_time),
            "success_rate": float(success_rate)
        }
    
    def update_metrics(self) -> None:
        """Update metrics with the latest data."""
        current_metrics = self.calculate_current_metrics()
        
        # Update metrics
        self.metrics["timestamps"].append(datetime.now().isoformat())
        self.metrics["avg_rating"].append(current_metrics["avg_rating"])
        self.metrics["command_accuracy"].append(current_metrics["command_accuracy"])
        self.metrics["response_time"].append(current_metrics["response_time"])
        self.metrics["success_rate"].append(current_metrics["success_rate"])
        
        # Save to file
        with open(METRICS_FILE, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a performance report."""
        if not self.metrics["timestamps"]:
            return "No metrics data available for reporting."
        
        # Get the most recent metrics
        latest = {
            "avg_rating": self.metrics["avg_rating"][-1],
            "command_accuracy": self.metrics["command_accuracy"][-1],
            "response_time": self.metrics["response_time"][-1],
            "success_rate": self.metrics["success_rate"][-1]
        }
        
        report = "Shell Agent Performance Report\n"
        report += "=============================\n"
        report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"Average Rating: {latest['avg_rating']:.2f} / 5.0\n"
        report += f"Command Accuracy: {latest['command_accuracy']:.2f} (proportion of correct commands)\n"
        report += f"Average Response Time: {latest['response_time']:.2f} seconds\n"
        report += f"Success Rate: {latest['success_rate']:.2f} (proportion of ratings >= 4)\n\n"
        
        # Add trend information if we have enough data points
        if len(self.metrics["timestamps"]) >= 3:
            rating_trend = self.metrics["avg_rating"][-1] - self.metrics["avg_rating"][-3]
            report += f"Rating Trend: {'↑' if rating_trend > 0 else '↓' if rating_trend < 0 else '→'} {abs(rating_trend):.2f}\n"
            
            accuracy_trend = self.metrics["command_accuracy"][-1] - self.metrics["command_accuracy"][-3]
            report += f"Accuracy Trend: {'↑' if accuracy_trend > 0 else '↓' if accuracy_trend < 0 else '→'} {abs(accuracy_trend):.2f}\n"
        
        return report
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot metrics over time."""
        if len(self.metrics["timestamps"]) < 2:
            print("Not enough data to plot metrics")
            return
        
        # Convert timestamps to datetime objects for plotting
        dates = [datetime.fromisoformat(ts) for ts in self.metrics["timestamps"]]
        
        # Create figure with multiple metrics
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Shell Agent Performance Metrics')
        
        # Plot each metric
        axs[0, 0].plot(dates, self.metrics["avg_rating"], 'b-')
        axs[0, 0].set_title('Average Rating')
        axs[0, 0].set_ylim([0, 5])
        
        axs[0, 1].plot(dates, self.metrics["command_accuracy"], 'g-')
        axs[0, 1].set_title('Command Accuracy')
        axs[0, 1].set_ylim([0, 1])
        
        axs[1, 0].plot(dates, self.metrics["response_time"], 'r-')
        axs[1, 0].set_title('Response Time (seconds)')
        
        axs[1, 1].plot(dates, self.metrics["success_rate"], 'c-')
        axs[1, 1].set_title('Success Rate')
        axs[1, 1].set_ylim([0, 1])
        
        # Format and save/show plot
        fig.autofmt_xdate()  # Rotate date labels
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

class ModelTrainer:
    """Provides training data and routines to improve the shell agent."""
    
    def __init__(self, data_collector: TrainingDataCollector):
        self.data_collector = data_collector
    
    def generate_training_examples(self) -> List[Dict[str, str]]:
        """Create training examples from feedback data."""
        training_examples = []
        
        for item in self.data_collector.feedback_data:
            # Only use entries with feedback and where a correct command was provided
            if item["feedback"] and item["feedback"]["correct_command"]:
                training_examples.append({
                    "user_request": item["user_request"],
                    "correct_command": item["feedback"]["correct_command"],
                    "incorrect_command": item["generated_command"],
                    "rating": item["feedback"]["rating"]
                })
        
        return training_examples
    
    def create_feedback_prompt(self, examples: List[Dict[str, str]]) -> str:
        """Create a prompt to help the model learn from feedback."""
        prompt = "Here are examples of user requests and the correct shell commands to use:\n\n"
        
        for ex in examples[:20]:  # Limit to 20 examples to avoid too large prompts
            prompt += f"User request: {ex['user_request']}\n"
            prompt += f"Incorrect command: {ex['incorrect_command']}\n"
            prompt += f"Correct command: {ex['correct_command']}\n\n"
        
        prompt += "Based on these examples, learn to generate more accurate shell commands for user requests."
        return prompt
    
    def suggest_model_improvements(self) -> str:
        """Generate suggestions for improving the shell agent based on feedback data."""
        examples = self.generate_training_examples()
        
        if not examples:
            return "No training examples available for improvement suggestions."
        
        # Create a prompt to analyze common mistakes
        analysis_prompt = "Analyze these examples of incorrect and correct shell commands:\n\n"
        
        for ex in examples[:10]:  # Limit to 10 examples
            analysis_prompt += f"User request: {ex['user_request']}\n"
            analysis_prompt += f"Incorrect command: {ex['incorrect_command']}\n"
            analysis_prompt += f"Correct command: {ex['correct_command']}\n\n"
        
        analysis_prompt += "What patterns of mistakes do you see? Provide specific recommendations for improving the shell command generation."
        
        # Get improvement suggestions from the LLM
        suggestions = call_deepseek(analysis_prompt)
        return suggestions
    
    def save_training_data(self) -> str:
        """Save training examples to a file for future model training."""
        examples = self.generate_training_examples()
        
        if not examples:
            return "No training examples available to save."
        
        # Save to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(MODELS_DIR, f"training_data_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(examples, f, indent=2)
        
        return f"Saved {len(examples)} training examples to {filename}"

def main():
    """Main function to run the training agent."""
    # Create instances
    collector = TrainingDataCollector()
    evaluator = PerformanceEvaluator(collector)
    trainer = ModelTrainer(collector)
    
    print("Shell Agent Training System")
    print("==========================")
    print("1. Collect training data")
    print("2. Show performance metrics")
    print("3. Generate improvement suggestions")
    print("4. Save training data")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            # Simulated data collection (in a real system, this would come from actual interactions)
            user_request = input("Enter a sample user request: ")
            command = input("What command was generated? ")
            result = execute_shell_command(command)
            
            collector.save_interaction(user_request, command, result, 1.0)
            
            # Get feedback
            rating = int(input("Rate the command (1-5): "))
            correct = input("Enter the correct command (if different) or leave empty: ")
            correct_command = correct if correct else None
            comments = input("Any comments? ")
            
            collector.add_user_feedback(len(collector.session_data)-1, rating, correct_command, comments)
            collector.save_feedback_data()
            evaluator.update_metrics()
            
            print("Feedback saved!")
            
        elif choice == "2":
            # Show metrics
            print(evaluator.generate_report())
            plot_choice = input("Generate plot? (y/n): ")
            if plot_choice.lower() == 'y':
                evaluator.plot_metrics()
            
        elif choice == "3":
            # Generate improvement suggestions
            print("Generating improvement suggestions...")
            suggestions = trainer.suggest_model_improvements()
            print("\nImprovement Suggestions:")
            print("------------------------")
            print(suggestions)
            
        elif choice == "4":
            # Save training data
            result = trainer.save_training_data()
            print(result)
            
        elif choice == "5":
            print("Exiting training agent.")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
