#!/usr/bin/env python3
"""
mcp_server_stdio.py — Stdio-compatible MCP server wrapper for GitHub repository auditing
This file provides stdio-based MCP server functionality that works with MCPServerStdio.
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import the FastMCP server functions
from mcp_server import (
    fetch_files, list_available_files, analyze_file, suggest_refactor, 
    generate_diff, display_fetched_files, display_diffs, get_session_data, 
    create_pull_request, analyze_security_vulnerabilities, benchmark_performance,
    run_tests_and_rollback, rank_refactoring_opportunities
)

# Setup
load_dotenv()

class MCPStdioServer:
    """Stdio-based MCP server that wraps FastMCP functions"""
    
    def __init__(self):
        self.tools = {
            "fetch_files": fetch_files,
            "list_available_files": list_available_files,
            "analyze_file": analyze_file,
            "suggest_refactor": suggest_refactor,
            "generate_diff": generate_diff,
            "display_fetched_files": display_fetched_files,
            "display_diffs": display_diffs,
            "get_session_data": get_session_data,
            "create_pull_request": create_pull_request,
            "analyze_security_vulnerabilities": analyze_security_vulnerabilities,
            "benchmark_performance": benchmark_performance,
            "run_tests_and_rollback": run_tests_and_rollback,
            "rank_refactoring_opportunities": rank_refactoring_opportunities
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            # Handle case where request_id might be None
            if request_id is None:
                request_id = 1
            
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "github-refactor-server",
                            "version": "1.0.0"
                        }
                    }
                }
            
            elif method == "notifications/initialized":
                # Handle initialized notification (no response needed)
                return None
            
            elif method == "tools/list":
                tools_list = []
                for tool_name, tool_func in self.tools.items():
                    # Define proper input schemas for each tool
                    input_schema = self._get_tool_schema(tool_name)
                    tools_list.append({
                        "name": tool_name,
                        "description": tool_func.__doc__ or f"Execute {tool_name}",
                        "inputSchema": input_schema
                    })
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": tools_list
                    }
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                if tool_name in self.tools:
                    tool_func = self.tools[tool_name]
                    
                    # Call the tool function with proper arguments
                    try:
                        # Handle special cases for function parameter mismatches
                        if tool_name == "suggest_refactor" and "filename" in tool_args and "files" in tool_args:
                            # Convert filename + files to file_content
                            filename = tool_args["filename"]
                            files = tool_args["files"]
                            if filename in files:
                                tool_args = {"file_content": files[filename]}
                            else:
                                tool_args = {"file_content": ""}
                        
                        # Check for missing required parameters
                        if tool_name == "analyze_file" and "files" not in tool_args:
                            return {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32602,
                                    "message": f"Missing required parameter 'files' for analyze_file tool. Please provide both 'filename' and 'files' parameters."
                                }
                            }
                        
                        # Get function signature to match parameters
                        import inspect
                        sig = inspect.signature(tool_func)
                        bound_args = sig.bind(**tool_args)
                        bound_args.apply_defaults()
                        
                        if asyncio.iscoroutinefunction(tool_func):
                            result = await tool_func(*bound_args.args, **bound_args.kwargs)
                        else:
                            result = tool_func(*bound_args.args, **bound_args.kwargs)
                        
                        # Properly serialize the result
                        if isinstance(result, dict):
                            result_text = json.dumps(result)
                        else:
                            result_text = str(result)
                        
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": result_text
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": f"Tool execution error: {str(e)}"
                            }
                        }
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Tool '{tool_name}' not found"
                        }
                    }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not found"
                    }
                }
                
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    def _get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get input schema for a specific tool"""
        schemas = {
            "fetch_files": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "GitHub repository URL"
                    }
                },
                "required": ["repo_url"]
            },
            "list_available_files": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "analyze_file": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to analyze"
                    },
                    "files": {
                        "type": "object",
                        "description": "Dictionary of files"
                    }
                },
                "required": ["filename", "files"]
            },
            "suggest_refactor": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to refactor"
                    },
                    "files": {
                        "type": "object",
                        "description": "Dictionary of files"
                    }
                },
                "required": ["filename", "files"]
            },
            "generate_diff": {
                "type": "object",
                "properties": {
                    "original": {
                        "type": "string",
                        "description": "Original code"
                    },
                    "optimized": {
                        "type": "string",
                        "description": "Optimized code"
                    }
                },
                "required": ["original", "optimized"]
            },
            "display_fetched_files": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "display_diffs": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "get_session_data": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "create_pull_request": {
                "type": "object",
                "properties": {
                    "branch_name": {
                        "type": "string",
                        "description": "Name of the branch"
                    },
                    "changes": {
                        "type": "object",
                        "description": "Dictionary of file changes"
                    },
                    "title": {
                        "type": "string",
                        "description": "Pull request title"
                    }
                },
                "required": ["branch_name", "changes", "title"]
            },
            "analyze_security_vulnerabilities": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "object",
                        "description": "Dictionary of files to analyze"
                    }
                },
                "required": ["files"]
            },
            "benchmark_performance": {
                "type": "object",
                "properties": {
                    "original_code": {
                        "type": "string",
                        "description": "Original code to benchmark"
                    },
                    "optimized_code": {
                        "type": "string",
                        "description": "Optimized code to benchmark"
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Number of benchmark iterations",
                        "default": 1000
                    }
                },
                "required": ["original_code", "optimized_code"]
            },
            "run_tests_and_rollback": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "object",
                        "description": "Original files"
                    },
                    "changes": {
                        "type": "object",
                        "description": "Proposed changes"
                    }
                },
                "required": ["files", "changes"]
            },
            "rank_refactoring_opportunities": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "object",
                        "description": "Dictionary of files to analyze"
                    }
                },
                "required": ["files"]
            }
        }
        
        return schemas.get(tool_name, {
            "type": "object",
            "properties": {},
            "required": []
        })
    
    async def run(self):
        """Run the stdio server"""
        print("🚀 Starting GitHub Refactor MCP Server (stdio)...", file=sys.stderr)
        
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                
                if response is not None:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError:
                continue
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                break

async def main():
    """Main entry point"""
    server = MCPStdioServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
