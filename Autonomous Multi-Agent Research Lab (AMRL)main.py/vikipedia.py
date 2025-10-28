#!/usr/bin/env python3
"""
Fast MCP Server for Wikipedia Integration
Provides Wikipedia search capabilities as MCP tools for AMRL agents
"""

import sys
from typing import Dict, Any, List, Optional
import wikipediaapi
from functools import lru_cache
from mcp.server.fastmcp import FastMCP

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(
    language='en', 
    user_agent='AMRL-MCP-Server/1.0 (https://github.com/amrl-research; contact: research@amrl.ai)'
)

# Create FastMCP server
mcp = FastMCP("Wikipedia MCP Server")

@mcp.tool()
@lru_cache(maxsize=100)
def wikipedia_search(query: str, sentences: int = 3) -> str:
    """
    Search Wikipedia for information about a given query.
    
    Args:
        query: The search term or topic to look up
        sentences: Number of sentences to return (default: 3)
        
    Returns:
        A concise summary from Wikipedia
    """
    try:
        page = wiki.page(query)
        if not page.exists():
            return f"❌ No Wikipedia page found for '{query}'."
        
        summary = page.summary.split('. ')
        short_summary = '. '.join(summary[:sentences]) + '.'
        
        return f"📚 Wikipedia Summary for '{query}':\n{short_summary}"
        
    except Exception as e:
        return f"❌ Error searching Wikipedia for '{query}': {str(e)}"

@mcp.tool()
def wikipedia_page_info(query: str) -> str:
    """
    Get detailed information about a Wikipedia page.
    
    Args:
        query: The search term or topic to look up
        
    Returns:
        Structured information including title, summary, URL, and categories
    """
    try:
        page = wiki.page(query)
        if not page.exists():
            return f"❌ No Wikipedia page found for '{query}'."
        
        # Get page information
        info = {
            "title": page.title,
            "summary": page.summary[:1000] + "..." if len(page.summary) > 1000 else page.summary,
            "url": page.fullurl,
            "categories": list(page.categories.keys())[:10],  # Limit categories
            "word_count": len(page.summary.split())
        }
        
        # Format as readable text
        result = f"""📖 Wikipedia Page Information for '{query}':

🏷️ Title: {info['title']}
📝 Summary: {info['summary']}
🔗 URL: {info['url']}
📊 Word Count: {info['word_count']}
🏷️ Categories: {', '.join(info['categories'][:5])}"""
        
        if len(info['categories']) > 5:
            result += f"\n... and {len(info['categories']) - 5} more categories"
        
        return result
        
    except Exception as e:
        return f"❌ Error getting Wikipedia page info for '{query}': {str(e)}"

@mcp.tool()
def wikipedia_search_multiple(queries: List[str], sentences: int = 2) -> str:
    """
    Search Wikipedia for multiple topics at once.
    
    Args:
        queries: List of search terms or topics
        sentences: Number of sentences per topic (default: 2)
        
    Returns:
        Combined summaries for all topics
    """
    try:
        results = []
        for query in queries:
            summary = wikipedia_search(query, sentences)
            results.append(summary)
        
        return f"🔍 Multiple Wikipedia Search Results:\n\n" + "\n\n".join(results)
        
    except Exception as e:
        return f"❌ Error in multiple Wikipedia search: {str(e)}"

@mcp.tool()
def wikipedia_related_topics(query: str, max_topics: int = 5) -> str:
    """
    Find Wikipedia pages related to a given topic.
    
    Args:
        query: The main topic to find related pages for
        max_topics: Maximum number of related topics to return
        
    Returns:
        List of related Wikipedia topics
    """
    try:
        page = wiki.page(query)
        if not page.exists():
            return f"❌ No Wikipedia page found for '{query}'."
        
        # Get links from the page
        links = list(page.links.keys())[:max_topics]
        
        if not links:
            return f"❌ No related topics found for '{query}'."
        
        result = f"🔗 Related Wikipedia Topics for '{query}':\n"
        for i, link in enumerate(links, 1):
            result += f"{i}. {link}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error finding related topics for '{query}': {str(e)}"

@mcp.tool()
def wikipedia_category_search(category: str, max_pages: int = 5) -> str:
    """
    Search Wikipedia pages within a specific category.
    
    Args:
        category: The Wikipedia category to search in
        max_pages: Maximum number of pages to return
        
    Returns:
        List of Wikipedia pages in the category
    """
    try:
        # Note: This is a simplified implementation
        # Real category search would require more complex Wikipedia API usage
        
        # For now, we'll search for the category as a regular page
        page = wiki.page(category)
        if not page.exists():
            return f"❌ No Wikipedia category found for '{category}'."
        
        # Get some links as examples
        links = list(page.links.keys())[:max_pages]
        
        result = f"📂 Wikipedia Pages in Category '{category}':\n"
        for i, link in enumerate(links, 1):
            result += f"{i}. {link}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error searching category '{category}': {str(e)}"

# Create a wrapper class for backward compatibility
class WikipediaMCPServer:
    """Wrapper class for backward compatibility with main.py"""
    
    def __init__(self):
        self.name = "wikipedia-mcp-server"
        self.version = "1.0.0"
    
    def wikipedia_search(self, query: str, sentences: int = 3) -> str:
        return wikipedia_search(query, sentences)
    
    def wikipedia_page_info(self, query: str) -> str:
        return wikipedia_page_info(query)
    
    def wikipedia_search_multiple(self, queries: List[str], sentences: int = 2) -> str:
        return wikipedia_search_multiple(queries, sentences)
    
    def wikipedia_related_topics(self, query: str, max_topics: int = 5) -> str:
        return wikipedia_related_topics(query, max_topics)
    
    def wikipedia_category_search(self, category: str, max_pages: int = 5) -> str:
        return wikipedia_category_search(category, max_pages)

# Create server instance for import
wikipedia_server = WikipediaMCPServer()

# Server configuration
server_config = {
    "name": "wikipedia-mcp-server",
    "version": "1.0.0",
    "description": "Fast MCP server for Wikipedia integration with AMRL system",
    "capabilities": {
        "tools": True,
        "resources": False,
        "prompts": False
    }
}

if __name__ == "__main__":
    # Run the FastMCP server using stdio
    mcp.run()
