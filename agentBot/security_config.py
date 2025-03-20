"""
Security configuration for the Shell Agent.
This file contains patterns and rules to identify potentially dangerous commands.
"""

# Dangerous command patterns that should be blocked
DANGEROUS_COMMAND_PATTERNS = [
    # System destruction commands
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+/*",
    r"rm\s+(-r|--recursive)\s+(-f|--force)\s+/",
    
    # Filesystem manipulation commands
    r"mkfs\.[a-zA-Z0-9]+\s+/dev/[a-zA-Z0-9]+",
    r"dd\s+if=.*\s+of=/dev/[a-zA-Z0-9]+",
    
    # Denial of service
    r"^\s*:(){ :\|: & };:",  # Fork bomb
    r"while true; do",  # Infinite loops
    
    # Permission changes
    r"chmod\s+-R\s+777\s+/",
    r"chmod\s+777\s+/etc/shadow",
    
    # Disk operations
    r"> /dev/[a-zA-Z0-9]+",
    r"/dev/null > /dev/[a-zA-Z0-9]+",
    
    # Remote code execution
    r"wget.+\|\s*bash",
    r"curl.+\|\s*bash",
    r"curl.+\|\s*sh",
    
    # Network flooding
    r"ping -f",
]

# Commands that require confirmation before execution
CONFIRMATION_REQUIRED_PATTERNS = [
    r"rm\s+-r",
    r"shutdown",
    r"reboot",
    r"halt",
    r"dd",
    r"fdisk",
    r"mkfs",
    r"mount",
    r"umount",
    r"passwd",
    r"chown\s+-R",
    r"chmod\s+-R",
]

def is_dangerous_command(command):
    """Check if a command matches any of the dangerous patterns."""
    import re
    command = command.lower()
    
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, command):
            return True
    return False

def requires_confirmation(command):
    """Check if a command requires confirmation before execution."""
    import re
    command = command.lower()
    
    for pattern in CONFIRMATION_REQUIRED_PATTERNS:
        if re.search(pattern, command):
            return True
    return False
