import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Configuration manager for the script optimizer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from files and environment variables.
        
        Args:
            config_path: Path to JSON configuration file. If not provided, 
                         will check for default config.json in the same directory.
        """
        self.config_data = {}
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        
        # Load from JSON config file if it exists
        self._load_from_file()
        
        # Override with environment variables (higher priority)
        self._load_from_env()
    
    def _load_from_file(self) -> None:
        """Load configuration from JSON file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    self.config_data = json.load(f)
                print(f"Loaded configuration from {self.config_path}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading configuration file: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Map environment variables to config keys
        env_map = {
            "OPENAI_API_KEY": "openai_api_key", 
            "DEEPSEEK_API_KEY": "deepseek_api_key",
            "DEFAULT_LLM_PROVIDER": "default_llm_provider",
            "DEFAULT_MODEL_NAME": "default_model_name",
            "MAX_ITERATIONS": "max_iterations",
        }
        
        # Update config with environment variables if they exist
        for env_var, config_key in env_map.items():
            if os.environ.get(env_var):
                self.config_data[config_key] = os.environ.get(env_var)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        return self.config_data.get(key, default)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider.
        
        Args:
            provider: LLM provider name ('openai' or 'deepseek')
            
        Returns:
            API key if available, None otherwise
        """
        if provider.lower() == "openai":
            return self.get("openai_api_key")
        elif provider.lower() == "deepseek":
            return self.get("deepseek_api_key")
        return None
    
    @classmethod
    def create_default_config(cls, output_path: Optional[str] = None) -> None:
        """Create a default configuration file template.
        
        Args:
            output_path: Path to save the configuration file. If not provided,
                         uses the default config path.
        """
        default_config = {
            "openai_api_key": "",
            "deepseek_api_key": "",
            "default_llm_provider": "openai",
            "default_model_name": "gpt-3.5-turbo",
            "max_iterations": 5,
            "output_directory": "./results"
        }
        
        path = output_path or DEFAULT_CONFIG_PATH
        with open(path, "w") as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Created default configuration template at {path}")
        print("Please edit this file to add your API keys and preferences.")


# Global configuration instance
config = Config()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage configuration for the Excel script optimizer")
    parser.add_argument("--create-config", action="store_true", help="Create a default configuration file")
    parser.add_argument("--output", help="Path for the created configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        Config.create_default_config(args.output)
