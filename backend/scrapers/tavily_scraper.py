"""
Tavily API scraper implementation.
"""
import requests
from typing import List, Dict, Any
import logging
from .base_scraper import BaseScraper
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TavilyScraper(BaseScraper):
    """Scraper using Tavily API."""
    
    def __init__(self):
        super().__init__('tavily')
        self.api_key = Config.TAVILY_API_KEY
        self.base_url = 'https://api.tavily.com/search'
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search using Tavily API.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.api_key:
            logger.error("Tavily API key not configured")
            return []
        
        try:
            logger.info(f"Searching Tavily for: {query}")
            
            payload = {
                'api_key': self.api_key,
                'query': query,
                'search_depth': 'advanced',
                'max_results': top_k,
                'include_answer': True,
                'include_raw_content': True
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=Config.REQUEST_TIMEOUT * 2
            )
            response.raise_for_status()
            data = response.json()
            
            # Store raw data
            self.raw_data = data
            
            results = []
            for item in data.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'description': item.get('content', ''),
                    'snippet': item.get('content', ''),
                    'score': item.get('score', 0),
                    'raw_content': item.get('raw_content', ''),
                    'published_date': item.get('published_date', '')
                })
            
            # Add answer if available
            if 'answer' in data:
                self.answer = data['answer']
            
            self.results = results
            logger.info(f"Tavily search completed: {len(self.results)} results")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in Tavily search: {e}")
            return []


if __name__ == "__main__":
    
    # Example usage: run this file directly
    scraper = TavilyScraper()

    if not scraper.api_key:
        print("Please set your Tavily API key in Config.TAVILY_API_KEY")
    else:
        query = "Best places to visit in Japan"
        top_k = 5
        output_dir = Config.OUTPUT_DIR

        # Run the search
        results = scraper.search(query, top_k=top_k)

        if results:
            # Save both raw and cleaned results
            filepaths = scraper.save_both(query, output_dir)
            print(f"Raw data saved to: {filepaths['raw_file']}")
            print(f"Cleaned results saved to: {filepaths['cleaned_file']}")

            # Print summary
            summary = scraper.get_metadata_summary()
            print("Summary:", summary)

            # Optional: print top results
            print(f"Top {top_k} results for query: '{query}'")
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} - {result['url']}")
