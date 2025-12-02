"""
LLM Service
Handles local language model interactions for synthesis and analysis
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        # In production, initialize your local LLM here
        # Options: Ollama, llama.cpp, transformers, etc.
        self.model_available = False
        self.model_name = "llama2"  # or any local model
    
    def is_available(self) -> bool:
        """Check if local LLM is available"""
        return self.model_available
    
    async def synthesize_security_results(
        self,
        query: str,
        github_data: Dict[str, Any],
        vulnerabilities: List[Dict[str, Any]],
        graph_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize security research results using LLM
        
        Args:
            query: Original user query
            github_data: Data from GitHub search
            vulnerabilities: Vulnerability data from OSV/NVD
            graph_analysis: Call graph analysis with PageRank/CheiRank
        """
        try:
            # Build context for LLM
            context = self._build_security_context(
                query, github_data, vulnerabilities, graph_analysis
            )
            
            # For MVP without local LLM, provide rule-based synthesis
            if not self.model_available:
                return self._rule_based_security_synthesis(
                    query, github_data, vulnerabilities, graph_analysis
                )
            
            # TODO: Call local LLM with context
            # response = await self._call_local_llm(context)
            
            return {
                "summary": "LLM synthesis would appear here",
                "key_findings": [],
                "recommendations": []
            }
        
        except Exception as e:
            logger.error(f"Error in LLM synthesis: {str(e)}")
            return {"error": str(e)}
    
    async def synthesize_healthcare_results(
        self,
        query: str,
        pubmed_data: Dict[str, Any],
        local_docs: List[Dict[str, Any]],
        citation_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize healthcare search results using LLM"""
        try:
            if not self.model_available:
                return self._rule_based_healthcare_synthesis(
                    query, pubmed_data, local_docs, citation_graph
                )
            
            # TODO: Call local LLM
            return {
                "summary": "LLM synthesis would appear here",
                "clinical_recommendations": [],
                "evidence_level": "Not evaluated"
            }
        
        except Exception as e:
            logger.error(f"Error in healthcare synthesis: {str(e)}")
            return {"error": str(e)}
    
    async def generate_fuzzing_strategy(
        self,
        graph_analysis: Dict[str, Any],
        vulnerabilities: List[Dict[str, Any]],
        focus_area: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate fuzzing strategy recommendations"""
        try:
            if not self.model_available:
                return self._rule_based_fuzzing_strategy(
                    graph_analysis, vulnerabilities, focus_area
                )
            
            # TODO: Call local LLM
            return {
                "strategy": "LLM-generated strategy",
                "targets": [],
                "harness_suggestions": []
            }
        
        except Exception as e:
            logger.error(f"Error generating fuzzing strategy: {str(e)}")
            return {"error": str(e)}
    
    async def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text using local LLM"""
        if not self.model_available:
            # Simple extractive summary (first N characters)
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # TODO: Call local LLM for abstractive summary
        return "Summary would appear here"
    
    async def answer_question(self, question: str, context: str) -> str:
        """Answer question using context"""
        if not self.model_available:
            return "Local LLM not available. Unable to answer question."
        
        # TODO: Call local LLM with question and context
        return "Answer would appear here"
    
    def _build_security_context(
        self,
        query: str,
        github_data: Dict[str, Any],
        vulnerabilities: List[Dict[str, Any]],
        graph_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """Build context string for LLM"""
        context_parts = [
            f"Query: {query}",
            f"\nGitHub Results: {len(github_data.get('repos', []))} repositories found",
            f"Vulnerabilities: {len(vulnerabilities)} CVEs identified"
        ]
        
        if graph_analysis:
            context_parts.append(
                f"Graph Analysis: {graph_analysis.get('nodes', 0)} nodes, "
                f"{graph_analysis.get('edges', 0)} edges"
            )
        
        return "\n".join(context_parts)
    
    def _rule_based_security_synthesis(
        self,
        query: str,
        github_data: Dict[str, Any],
        vulnerabilities: List[Dict[str, Any]],
        graph_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Rule-based synthesis when LLM is not available"""
        
        repos = github_data.get("repos", [])
        issues = github_data.get("issues", [])
        
        # Extract key findings
        key_findings = []
        
        if repos:
            top_repo = repos[0]
            key_findings.append(
                f"Most relevant repository: {top_repo['full_name']} "
                f"({top_repo['stars']} stars)"
            )
        
        if vulnerabilities:
            high_severity = [v for v in vulnerabilities if v.get("severity") in ["HIGH", "CRITICAL"]]
            if high_severity:
                key_findings.append(
                    f"Found {len(high_severity)} high/critical severity vulnerabilities"
                )
        
        if graph_analysis and "critical_nodes" in graph_analysis:
            critical = graph_analysis["critical_nodes"][:3]
            if critical:
                key_findings.append(
                    f"Critical code paths identified: {', '.join([n['node'] for n in critical])}"
                )
        
        # Generate recommendations
        recommendations = []
        
        if graph_analysis and "top_pagerank_nodes" in graph_analysis:
            top_pr = graph_analysis["top_pagerank_nodes"][:3]
            recommendations.append({
                "priority": "HIGH",
                "action": "Focus fuzzing on heavily-used functions",
                "targets": [n["node"] for n in top_pr],
                "rationale": "These functions have high PageRank, indicating many dependencies"
            })
        
        if vulnerabilities:
            recommendations.append({
                "priority": "CRITICAL",
                "action": "Review and patch known vulnerabilities",
                "cves": [v["id"] for v in vulnerabilities[:5]],
                "rationale": "Known CVEs exist in related code"
            })
        
        if issues:
            security_issues = [i for i in issues if "security" in [l.lower() for l in i.get("labels", [])]]
            if security_issues:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "Review open security issues",
                    "count": len(security_issues),
                    "rationale": "Active security discussions in the community"
                })
        
        return {
            "summary": f"Analysis of '{query}' found {len(repos)} relevant repositories, "
                      f"{len(issues)} related issues, and {len(vulnerabilities)} known vulnerabilities.",
            "key_findings": key_findings,
            "recommendations": recommendations,
            "confidence": "MEDIUM (rule-based analysis)",
            "note": "This is a rule-based synthesis. Enable local LLM for deeper analysis."
        }
    
    def _rule_based_healthcare_synthesis(
        self,
        query: str,
        pubmed_data: Dict[str, Any],
        local_docs: List[Dict[str, Any]],
        citation_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rule-based healthcare synthesis"""
        
        articles = pubmed_data.get("articles", [])
        
        # Extract key papers
        top_papers = []
        if citation_graph and "top_papers" in citation_graph:
            top_papers = citation_graph["top_papers"][:3]
        
        recommendations = []
        
        if articles:
            recommendations.append({
                "level": "Review",
                "action": f"Examine {len(articles)} relevant articles",
                "top_journals": list(set([a.get("journal", "") for a in articles[:5]])),
                "publication_range": f"{articles[-1].get('publication_date', '')} to {articles[0].get('publication_date', '')}"
            })
        
        # Extract common MeSH terms
        all_mesh = []
        for article in articles:
            all_mesh.extend(article.get("mesh_terms", []))
        
        common_mesh = {}
        for term in all_mesh:
            common_mesh[term] = common_mesh.get(term, 0) + 1
        
        top_mesh = sorted(common_mesh.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "summary": f"Found {len(articles)} relevant medical articles on '{query}'",
            "key_topics": [term for term, _ in top_mesh],
            "top_cited_papers": top_papers,
            "clinical_recommendations": recommendations,
            "evidence_level": "Multiple studies available",
            "note": "This is a rule-based synthesis. Enable local LLM for clinical decision support."
        }
    
    def _rule_based_fuzzing_strategy(
        self,
        graph_analysis: Dict[str, Any],
        vulnerabilities: List[Dict[str, Any]],
        focus_area: Optional[str]
    ) -> Dict[str, Any]:
        """Rule-based fuzzing strategy generation"""
        
        strategy = {
            "focus_area": focus_area or "general",
            "approach": "multi-target",
            "tools": ["syzkaller", "AFL++", "libFuzzer"],
            "targets": [],
            "harness_suggestions": []
        }
        
        # Prioritize based on graph analysis
        if graph_analysis:
            critical_nodes = graph_analysis.get("critical_nodes", [])
            top_cheirank = graph_analysis.get("top_cheirank_nodes", [])
            
            # High CheiRank nodes = functions that call many others
            # Good fuzzing targets for coverage
            for node in top_cheirank[:5]:
                strategy["targets"].append({
                    "function": node["node"],
                    "score": node["score"],
                    "reason": "High CheiRank - calls many functions, good for coverage",
                    "priority": "HIGH"
                })
            
            # Critical nodes = high in both rankings
            for node in critical_nodes[:3]:
                if node not in strategy["targets"]:
                    strategy["harness_suggestions"].append({
                        "function": node["node"],
                        "approach": "directed",
                        "reason": "Critical node - widely used and influential"
                    })
        
        # Add vulnerability-based targets
        for vuln in vulnerabilities[:3]:
            strategy["targets"].append({
                "vulnerability": vuln.get("id"),
                "severity": vuln.get("severity"),
                "reason": "Known vulnerability - verify patch and find similar issues",
                "priority": "CRITICAL"
            })
        
        strategy["recommendation"] = (
            f"Focus fuzzing on {len(strategy['targets'])} high-priority targets. "
            f"Start with syzkaller for system call fuzzing, then use AFL++ for "
            f"specific function harnesses."
        )
        
        return strategy
