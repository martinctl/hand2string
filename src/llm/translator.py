# Post-process raw ASL word sequences into fluent English.
# Backends: local Llama 3 via Ollama, or a hosted API.


def translate(words: list[str], backend: str = "ollama", model: str = "llama3:8b") -> str:
    raise NotImplementedError
