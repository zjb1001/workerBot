"""
Command history module for the Shell Agent.
Maintains a record of executed commands with timestamps and results.
"""

import json
import time
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class CommandHistory:
    def __init__(self, history_file: str = None):
        self.history: List[Dict[str, Any]] = []
        self.history_file = history_file or os.path.join(os.path.dirname(__file__), "command_history.json")
        self._load_history()
    
    def add_command(self, command: str, output: str, success: bool = True) -> None:
        """Add a command to the history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "success": success,
            "output": output[:500] if output else ""  # Limit output size
        }
        self.history.append(entry)
        self._save_history()
    
    def get_recent_commands(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent commands."""
        return self.history[-limit:] if self.history else []
    
    def get_formatted_history(self, limit: int = 10) -> str:
        """Get a formatted string of command history."""
        if not self.history:
            return "No commands have been executed yet."
        
        recent = self.get_recent_commands(limit)
        
        formatted = "Command History:\n"
        formatted += "--------------\n"
        for i, entry in enumerate(recent, 1):
            time_str = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            status = "✓" if entry["success"] else "✗"
            formatted += f"{i}. [{time_str}] {status} {entry['command']}\n"
        
        return formatted
    
    def search_history(self, keyword: str) -> List[Dict[str, Any]]:
        """Search command history for a keyword."""
        return [entry for entry in self.history if keyword in entry["command"]]
    
    def clear_history(self) -> None:
        """Clear the command history."""
        self.history = []
        self._save_history()
    
    def _save_history(self) -> None:
        """Save history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history[-100:], f)  # Keep only last 100 commands
        except Exception as e:
            print(f"Error saving command history: {e}")
    
    def _load_history(self) -> None:
        """Load history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"Error loading command history: {e}")
            self.history = []
