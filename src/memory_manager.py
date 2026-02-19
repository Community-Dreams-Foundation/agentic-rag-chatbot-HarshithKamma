import os

USER_MEMORY_FILE = "USER_MEMORY.md"
COMPANY_MEMORY_FILE = "COMPANY_MEMORY.md"

class MemoryManager:
    def __init__(self):
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        """Creates memory files if they don't exist."""
        for filename in [USER_MEMORY_FILE, COMPANY_MEMORY_FILE]:
            if not os.path.exists(filename):
                with open(filename, "w") as f:
                    f.write(f"# {filename.split('.')[0].replace('_', ' ')}\n\n")

    def _append_to_file(self, filename: str, content: str):
        """Appends a new bullet point to the specified file."""
        entry = f"- {content}\n"
        with open(filename, "a") as f:
            f.write(entry)

    def update_user_memory(self, fact: str):
        """Writes high-signal user fact to USER_MEMORY.md"""
        print(f"[Memory] Writing to User Memory: {fact}")
        self._append_to_file(USER_MEMORY_FILE, fact)

    def update_company_memory(self, learning: str):
        """Writes re-usable org learning to COMPANY_MEMORY.md"""
        print(f"[Memory] Writing to Company Memory: {learning}")
        self._append_to_file(COMPANY_MEMORY_FILE, learning)
