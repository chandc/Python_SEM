import os

try:
    import tomli
except ImportError:
    print("Error: 'tomli' module is not installed. Please install it using 'uv pip install tomli'.")
    raise

class Config:
    """
    A simple configuration wrapper for TOML input files.
    """
    def __init__(self, filepath="input.toml"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
            
        with open(filepath, "rb") as f:
            self._config = tomli.load(f)
            
    def get(self, section, key, default=None):
        """Safely fetch a key from a specific section."""
        if section not in self._config:
            return default
        return self._config[section].get(key, default)
        
    @property
    def raw(self):
        """Access the raw parsed TOML dictionary."""
        return self._config
