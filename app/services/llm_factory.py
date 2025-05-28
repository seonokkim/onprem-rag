from typing import Any, Dict, List, Type

import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from app.config.settings import get_settings


class LLMFactory:
    def __init__(self, provider_type: str):
        self.provider_type = provider_type
        self.settings = get_settings()
        
        if provider_type == "chat_model":
            self.model_settings = self.settings.chat_model
        elif provider_type == "embedding_model":
            self.model_settings = self.settings.embedding_model
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

        # Determine the actual LLM provider based on the model settings
        # Assuming base_url implies a local provider like Ollama (using OpenAI API compatibility)
        # If you had other providers like Anthropic, you would add checks here
        if self.model_settings.base_url:
             self.provider = "llama" # Treat local models as 'llama' for client initialization
        elif self.model_settings.api_key: # Example check for API based providers
             # This part might need adjustment based on how other providers are configured
             if "anthropic" in self.model_settings.api_key.lower(): # Simple check, adjust as needed
                 self.provider = "anthropic"
             else:
                 raise ValueError(f"Could not determine LLM provider from settings for type: {provider_type}")
        else:
             raise ValueError(f"Could not determine LLM provider from settings for type: {provider_type}")

        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        client_initializers = {
            "anthropic": lambda s: instructor.from_anthropic(
                Anthropic(api_key=s.api_key)
            ),
            "llama": lambda s: instructor.from_openai(
                OpenAI(
                    base_url=f"{s.base_url}/v1",
                    api_key="ollama",  # Ollama doesn't require a real API key
                ),
                mode=instructor.Mode.JSON,
            ),
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.model_settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.model_settings.default_model),
            "temperature": kwargs.get("temperature", self.model_settings.temperature),
            "max_retries": kwargs.get("max_retries", self.model_settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.model_settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create(**completion_params)
