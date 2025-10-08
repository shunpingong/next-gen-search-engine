"""
Base scraper class for all scraping implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
import json
from pathlib import Path


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.raw_data = None  # Store raw API response data
        
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a search with the given query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing search results
        """
        pass
    
    def save_raw_data(self, query: str, output_dir: Path) -> str:
        """
        Save raw API response data to a JSON file.
        
        Args:
            query: The search query used
            output_dir: Directory to save the output
            
        Returns:
            Path to the saved raw file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.name}_raw_{timestamp}.json"
        filepath = output_dir / filename
        
        raw_output_data = {
            'scraper': self.name,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'raw_data': self.raw_data
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_output_data, f, indent=2, ensure_ascii=False)
            
        return str(filepath)

    def save_results(self, query: str, output_dir: Path) -> str:
        """
        Save cleaned results to a JSON file.
        
        Args:
            query: The search query used
            output_dir: Directory to save the output
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.name}_cleaned_{timestamp}.json"
        filepath = output_dir / filename
        
        output_data = {
            'scraper': self.name,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(self.results),
            'results': self.results
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        return str(filepath)
    
    def save_both(self, query: str, output_dir: Path) -> Dict[str, str]:
        """
        Save both raw and cleaned data to separate JSON files.
        
        Args:
            query: The search query used
            output_dir: Directory to save the output
            
        Returns:
            Dictionary with paths to both saved files
        """
        raw_filepath = self.save_raw_data(query, output_dir)
        cleaned_filepath = self.save_results(query, output_dir)
        
        return {
            'raw_file': raw_filepath,
            'cleaned_file': cleaned_filepath
        }
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the scraped data."""
        if not self.results:
            return {}
            
        return {
            'total_results': len(self.results),
            'avg_title_length': sum(len(r.get('title', '')) for r in self.results) / len(self.results),
            'avg_description_length': sum(len(r.get('description', '')) for r in self.results) / len(self.results),
            'results_with_content': sum(1 for r in self.results if r.get('content')),
        }
