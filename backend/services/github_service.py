"""
GitHub API Service
Handles interactions with GitHub REST API for code search, issues, and repository analysis
"""

import os
import aiohttp
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GitHubService:
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.token = os.getenv("GITHUB_TOKEN")  # Set in .env file
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
    
    def is_configured(self) -> bool:
        """Check if GitHub token is configured"""
        return bool(self.token)
    
    async def search(
        self,
        query: str,
        search_type: str = "repositories",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Generic GitHub search
        
        Args:
            query: Search query
            search_type: 'repositories', 'issues', 'code', or 'commits'
            max_results: Maximum number of results
        """
        try:
            url = f"{self.base_url}/search/{search_type}"
            params = {
                "q": query,
                "per_page": min(max_results, 100),
                "sort": "stars" if search_type == "repositories" else "updated"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "total_count": data.get("total_count", 0),
                            "items": data.get("items", [])[:max_results]
                        }
                    else:
                        logger.error(f"GitHub API error: {response.status}")
                        return {"total_count": 0, "items": [], "error": await response.text()}
        
        except Exception as e:
            logger.error(f"Error in GitHub search: {str(e)}")
            return {"total_count": 0, "items": [], "error": str(e)}
    
    async def search_security_issues(
        self,
        query: str,
        domain: str = "linux-kernel",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for security-related issues and repositories
        
        Args:
            query: User query
            domain: Security domain (linux-kernel, networking, etc.)
            max_results: Maximum results per search type
        """
        # Build enhanced query
        security_keywords = ["vulnerability", "CVE", "security", "fuzzing", "syzkaller"]
        domain_keywords = {
            "linux-kernel": ["linux kernel", "torvalds/linux"],
            "networking": ["network driver", "net/", "socket"],
            "filesystem": ["filesystem", "fs/", "VFS"]
        }
        
        enhanced_query = f"{query} {' OR '.join(domain_keywords.get(domain, [domain]))}"
        
        # Search repositories
        repos = await self.search(
            query=f"{enhanced_query} {' OR '.join(security_keywords)}",
            search_type="repositories",
            max_results=max_results
        )
        
        # Search issues
        issues = await self.search(
            query=f"{enhanced_query} label:security OR label:vulnerability",
            search_type="issues",
            max_results=max_results
        )
        
        # Search code
        code = await self.search(
            query=f"{enhanced_query} extension:c OR extension:h",
            search_type="code",
            max_results=max_results
        )
        
        return {
            "query": query,
            "domain": domain,
            "repos": [self._format_repo(r) for r in repos.get("items", [])],
            "issues": [self._format_issue(i) for i in issues.get("items", [])],
            "code": [self._format_code(c) for c in code.get("items", [])]
        }
    
    async def get_repo_issues(
        self,
        owner: str,
        repo: str,
        labels: Optional[List[str]] = None,
        state: str = "open"
    ) -> List[Dict[str, Any]]:
        """Get issues from a specific repository"""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            params = {
                "state": state,
                "per_page": 100
            }
            if labels:
                params["labels"] = ",".join(labels)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [self._format_issue(issue) for issue in data]
                    else:
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching repo issues: {str(e)}")
            return []
    
    async def get_repo_files(
        self,
        owner: str,
        repo: str,
        path: str = "",
        pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get files from a repository"""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        files = []
                        
                        if isinstance(data, list):
                            for item in data:
                                if item["type"] == "file":
                                    if not pattern or item["name"].endswith(pattern.replace("*", "")):
                                        files.append({
                                            "name": item["name"],
                                            "path": item["path"],
                                            "size": item["size"],
                                            "download_url": item.get("download_url"),
                                            "url": item.get("html_url")
                                        })
                        
                        return files
                    else:
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching repo files: {str(e)}")
            return []
    
    async def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Get content of a specific file"""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Content is base64 encoded
                        import base64
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        return content
                    else:
                        return None
        
        except Exception as e:
            logger.error(f"Error fetching file content: {str(e)}")
            return None
    
    def extract_packages(self, github_results: Dict[str, Any]) -> List[str]:
        """Extract package names from GitHub search results"""
        packages = []
        
        for repo in github_results.get("repos", []):
            packages.append(f"{repo['owner']}/{repo['name']}")
        
        return packages
    
    def _format_repo(self, repo: Dict) -> Dict[str, Any]:
        """Format repository data"""
        return {
            "owner": repo["owner"]["login"],
            "name": repo["name"],
            "full_name": repo["full_name"],
            "description": repo.get("description", ""),
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "url": repo["html_url"],
            "language": repo.get("language"),
            "topics": repo.get("topics", []),
            "updated_at": repo.get("updated_at")
        }
    
    def _format_issue(self, issue: Dict) -> Dict[str, Any]:
        """Format issue data"""
        return {
            "id": issue["id"],
            "number": issue["number"],
            "title": issue["title"],
            "state": issue["state"],
            "labels": [label["name"] for label in issue.get("labels", [])],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "url": issue["html_url"],
            "body": issue.get("body", "")[:500]  # Truncate for brevity
        }
    
    def _format_code(self, code: Dict) -> Dict[str, Any]:
        """Format code search result"""
        return {
            "name": code["name"],
            "path": code["path"],
            "repository": code["repository"]["full_name"],
            "url": code["html_url"],
            "score": code.get("score", 0)
        }
