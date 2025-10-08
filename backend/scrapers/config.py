"""
Configuration loader for API keys and settings.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Configuration class to hold all API keys and settings."""
    
    # Google Custom Search API
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    # Tavily API
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    
    # Serper API
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    
    # SerpAPI
    SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
    
    # Firecrawl API
    FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
    
    # Scraping settings
    TOP_K_RESULTS = 10
    REQUEST_TIMEOUT = 10
    MAX_THREADS = 5
    
    # Output directory
    OUTPUT_DIR = Path(__file__).parent / 'output'
    
    @classmethod
    def validate(cls, services: list[str]) -> dict[str, bool]:
        """Validate if API keys are available for specified services."""
        validation = {}
        
        for service in services:
            if service == 'google':
                validation[service] = bool(cls.GOOGLE_API_KEY and cls.GOOGLE_SEARCH_ENGINE_ID)
            elif service == 'tavily':
                validation[service] = bool(cls.TAVILY_API_KEY)
            elif service == 'serper':
                validation[service] = bool(cls.SERPER_API_KEY)
            elif service == 'serpapi':
                validation[service] = bool(cls.SERPAPI_API_KEY)
            elif service == 'firecrawl':
                validation[service] = bool(cls.FIRECRAWL_API_KEY)
            elif service in ['duckduckgo', 'selenium']:
                validation[service] = True  # No API key needed
            else:
                validation[service] = False
                
        return validation
