"""LLM client abstraction for different LLM providers."""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    tokens_used: int
    model: str
    
    def __str__(self):
        return self.text


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate text from LLM.
        
        Args:
            prompt: User prompt/message
            temperature: Temperature for sampling (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system message
        
        Returns:
            LLMResponse with generated text
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
        """
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = model
            logger.info(f"✓ Initialized OpenAI client with model={model}")
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate text using OpenAI API."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens,
                model=self.model
            )
        
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def generate_streaming(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ):
        """Generate text with streaming."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            raise


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=api_key)
            self.model = model
            logger.info(f"✓ Initialized Anthropic client with model={model}")
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate text using Anthropic API."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return LLMResponse(
                text=response.content[0].text,
                tokens_used=response.usage.output_tokens,
                model=self.model
            )
        
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise


class LocalLLMClient(LLMClient):
    """Client for local LLMs via ollama or similar."""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize local LLM client.
        
        Args:
            model: Model to use
            base_url: Base URL for local LLM server
        """
        try:
            import requests
            self.model = model
            self.base_url = base_url
            self.requests = requests
            logger.info(f"✓ Initialized local LLM client with model={model}")
        except ImportError:
            raise ImportError("requests package required")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate text using local LLM."""
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )
            
            result = response.json()
            
            return LLMResponse(
                text=result.get("response", ""),
                tokens_used=len(result.get("response", "").split()),  # Approximate
                model=self.model
            )
        
        except Exception as e:
            logger.error(f"Local LLM error: {str(e)}")
            raise


def create_llm_client(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM client.
    
    Args:
        provider: "openai", "anthropic", or "local"
        api_key: API key for provider
        model: Model name/ID
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Initialized LLM client
    """
    import os
    
    if provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        model = model or os.getenv("LLM_MODEL", "gpt-4o")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        return OpenAIClient(api_key=api_key, model=model)
    
    elif provider == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        model = model or os.getenv("LLM_MODEL", "claude-3-opus-20240229")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")
        return AnthropicClient(api_key=api_key, model=model)
    
    elif provider == "local":
        model = model or "llama2"
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return LocalLLMClient(model=model, base_url=base_url)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
