import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage

from app.models.schemas import AgentType, AgentStatus

# Ensure environment variables are loaded
load_dotenv()

class BaseAgent:
    """Base class for all agents with LangChain LLM initialization using OpenRouter"""
    
    def __init__(self, agent_type: AgentType, system_prompt: str, temperature: float = 0.7):
        self.agent_type = agent_type
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.status = AgentStatus.IDLE
        
        # Initialize LLM with OpenRouter - using API key from .env file
        # First ensure .env is loaded (in case called before main.py)
        load_dotenv()
        # API key from backend/.env file
        api_key = os.getenv("OPENROUTER_API_KEY") or "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"
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Please check backend/.env file.")
        
        # Use the model specified in environment, or default to free model
        # Options: openrouter/polaris-alpha (free), openai/gpt-4o (paid), etc.
        # Default to polaris-alpha which is free and reliable
        model_name = os.getenv("OPENROUTER_MODEL", "openrouter/polaris-alpha")
        
        self.llm = ChatOpenAI(
            model=model_name,  # Use 'model' parameter, not 'model_name'
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=4096,  # Reasonable token limit
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/rover-ops",
                "X-Title": "Rover Ops"
            }
        )
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
    
    async def process(self, input_data: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input and return agent response
        Override in subclasses for specific behavior
        """
        self.status = AgentStatus.EXECUTING
        
        try:
            # Format prompt with context if provided
            messages = self.prompt_template.format_messages(
                input=input_data,
                context=context or {}
            )
            
            # Call LLM
            response = await self.llm.ainvoke(messages)
            
            result = {
                "agent_type": self.agent_type.value,
                "status": "success",
                "response": response.content,
                "raw_response": response
            }
            
            self.status = AgentStatus.IDLE
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {
                "agent_type": self.agent_type.value,
                "status": "error",
                "error": str(e)
            }
    
    def set_status(self, status: AgentStatus):
        """Update agent status"""
        self.status = status
    
    def get_status(self) -> AgentStatus:
        """Get current agent status"""
        return self.status

