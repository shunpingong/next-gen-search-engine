"""
OSV.dev API Service
Handles queries to Open Source Vulnerabilities database
"""

import aiohttp
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OSVService:
    def __init__(self):
        self.base_url = "https://api.osv.dev/v1"
    
    async def check_vulnerabilities(
        self,
        packages: List[str],
        ecosystem: str = "Linux"
    ) -> List[Dict[str, Any]]:
        """
        Check vulnerabilities for a list of packages
        
        Args:
            packages: List of package identifiers
            ecosystem: Package ecosystem (Linux, PyPI, npm, etc.)
        """
        all_vulns = []
        
        for package in packages:
            vulns = await self.query_package(package, ecosystem)
            if vulns:
                all_vulns.extend(vulns)
        
        return all_vulns
    
    async def query_package(
        self,
        package: str,
        ecosystem: str = "Linux"
    ) -> List[Dict[str, Any]]:
        """Query vulnerabilities for a single package"""
        try:
            url = f"{self.base_url}/query"
            payload = {
                "package": {
                    "name": package,
                    "ecosystem": ecosystem
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        vulns = data.get("vulns", [])
                        return [self._format_vulnerability(v, package) for v in vulns]
                    else:
                        logger.warning(f"OSV API returned {response.status} for {package}")
                        return []
        
        except Exception as e:
            logger.error(f"Error querying OSV for {package}: {str(e)}")
            return []
    
    async def get_vulnerability(self, vuln_id: str) -> Dict[str, Any]:
        """Get details of a specific vulnerability"""
        try:
            url = f"{self.base_url}/vulns/{vuln_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_vulnerability(data)
                    else:
                        return {}
        
        except Exception as e:
            logger.error(f"Error fetching vulnerability {vuln_id}: {str(e)}")
            return {}
    
    async def batch_query(self, queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Batch query multiple packages"""
        try:
            url = f"{self.base_url}/querybatch"
            payload = {"queries": queries}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"results": []}
        
        except Exception as e:
            logger.error(f"Error in batch query: {str(e)}")
            return {"results": []}
    
    def _format_vulnerability(self, vuln: Dict, package: str = "") -> Dict[str, Any]:
        """Format vulnerability data"""
        severity = "UNKNOWN"
        if "severity" in vuln:
            severity = vuln["severity"][0].get("score", "UNKNOWN") if vuln["severity"] else "UNKNOWN"
        elif "database_specific" in vuln:
            severity = vuln["database_specific"].get("severity", "UNKNOWN")
        
        return {
            "id": vuln.get("id", ""),
            "package": package,
            "summary": vuln.get("summary", ""),
            "details": vuln.get("details", ""),
            "severity": severity,
            "published": vuln.get("published", ""),
            "modified": vuln.get("modified", ""),
            "aliases": vuln.get("aliases", []),
            "references": [ref.get("url") for ref in vuln.get("references", [])],
            "affected": vuln.get("affected", [])
        }
