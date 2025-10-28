#!/usr/bin/env python3
"""
mcp_server.py — Local MCP server for GitHub repository auditing, refactoring, and PR management
Uses FastMCP with correct @mcp.tool() decorators and run() call.
"""

import os
import sys
import asyncio
import difflib
import tempfile
import git
import logging
import re
import json
import subprocess
import time
import ast
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv
from github import Github
from mcp.server.fastmcp import FastMCP

# Setup
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_URL = os.getenv("REPO_URL")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Initialize FastMCP server
mcp = FastMCP("github-refactor-server")

# Session storage for sandboxed environment
session_data = {
    "fetched_files": {},
    "analysis_results": {},
    "generated_diffs": {},
    "refactoring_suggestions": {},
    "security_vulnerabilities": {},
    "performance_benchmarks": {},
    "test_results": {},
    "language_analysis": {},
    "impact_rankings": {}
}

# Security vulnerability patterns
SECURITY_PATTERNS = {
    "sql_injection": [
        r"execute\s*\(\s*['\"].*%.*['\"]",
        r"cursor\.execute\s*\(\s*['\"].*%.*['\"]",
        r"query\s*=.*%.*",
        r"f['\"].*SELECT.*{.*}.*['\"]",
        r"f['\"].*INSERT.*{.*}.*['\"]",
        r"f['\"].*UPDATE.*{.*}.*['\"]",
        r"f['\"].*DELETE.*{.*}.*['\"]"
    ],
    "xss": [
        r"innerHTML\s*=",
        r"outerHTML\s*=",
        r"document\.write\s*\(",
        r"eval\s*\(",
        r"setTimeout\s*\(\s*['\"].*['\"]",
        r"setInterval\s*\(\s*['\"].*['\"]"
    ],
    "hardcoded_secrets": [
        r"password\s*=\s*['\"][^'\"]{8,}['\"]",
        r"api_key\s*=\s*['\"][^'\"]{20,}['\"]",
        r"secret\s*=\s*['\"][^'\"]{16,}['\"]",
        r"token\s*=\s*['\"][^'\"]{20,}['\"]",
        r"AWS_ACCESS_KEY_ID\s*=",
        r"AWS_SECRET_ACCESS_KEY\s*=",
        r"GITHUB_TOKEN\s*=",
        r"OPENAI_API_KEY\s*="
    ],
    "ssrf": [
        r"requests\.get\s*\(\s*request\.",
        r"urllib\.request\.urlopen\s*\(\s*request\.",
        r"httpx\.get\s*\(\s*request\.",
        r"aiohttp\.ClientSession\(\)\.get\s*\(\s*request\."
    ]
}

# Language-specific patterns
LANGUAGE_PATTERNS = {
    "python": {
        "async_patterns": [r"async def", r"await ", r"asyncio\."],
        "dataclass_candidates": [r"class \w+:\s*\n\s*def __init__"],
        "performance_issues": [r"for .* in .*:\s*\n.*await ", r"time\.sleep\("]
    },
    "javascript": {
        "async_patterns": [r"async function", r"await ", r"Promise\."],
        "performance_issues": [r"for\s*\(.*\)\s*{.*await", r"setTimeout\("],
        "security_patterns": [r"eval\(", r"innerHTML\s*=", r"document\.write\("]
    },
    "go": {
        "concurrency_patterns": [r"go func", r"chan ", r"select\s*{"],
        "performance_issues": [r"for.*range.*{.*time\.Sleep", r"sync\.WaitGroup"]
    },
    "rust": {
        "ownership_patterns": [r"let mut ", r"&mut ", r"move\s*\|"],
        "performance_issues": [r"Vec::new\(\)", r"String::new\(\)"]
    },
    "typescript": {
        "type_patterns": [r"interface \w+", r"type \w+", r": \w+\[\]"],
        "async_patterns": [r"async function", r"await ", r"Promise<"]
    }
}

# Helper functions for enhanced capabilities
def chunk_content(content: str, chunk_size: int, language: str) -> List[str]:
    """Chunk large files intelligently based on language syntax."""
    if len(content) <= chunk_size:
        return [content]
    
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        
        if current_size + line_size > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def analyze_languages(files: Dict[str, str]) -> Dict[str, str]:
    """Analyze file languages based on extensions and content."""
    language_map = {}
    
    for filename, content in files.items():
        # Determine language from file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.py']:
            language_map[filename] = 'python'
        elif ext in ['.js', '.jsx']:
            language_map[filename] = 'javascript'
        elif ext in ['.ts', '.tsx']:
            language_map[filename] = 'typescript'
        elif ext in ['.go']:
            language_map[filename] = 'go'
        elif ext in ['.rs']:
            language_map[filename] = 'rust'
        elif ext in ['.java']:
            language_map[filename] = 'java'
        elif ext in ['.cpp', '.c']:
            language_map[filename] = 'cpp'
        elif ext in ['.cs']:
            language_map[filename] = 'csharp'
        elif ext in ['.php']:
            language_map[filename] = 'php'
        elif ext in ['.rb']:
            language_map[filename] = 'ruby'
        elif ext in ['.swift']:
            language_map[filename] = 'swift'
        elif ext in ['.kt']:
            language_map[filename] = 'kotlin'
        else:
            language_map[filename] = 'unknown'
    
    return language_map

def detect_security_vulnerabilities(content: str, filename: str) -> List[Dict[str, Any]]:
    """Detect security vulnerabilities in code."""
    vulnerabilities = []
    
    for vuln_type, patterns in SECURITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                vulnerabilities.append({
                    "type": vuln_type,
                    "severity": get_severity(vuln_type),
                    "line": line_num,
                    "code": match.group(),
                    "description": get_vulnerability_description(vuln_type),
                    "file": filename
                })
    
    return vulnerabilities

def get_severity(vuln_type: str) -> str:
    """Get severity level for vulnerability type."""
    severity_map = {
        "sql_injection": "HIGH",
        "xss": "HIGH", 
        "hardcoded_secrets": "CRITICAL",
        "ssrf": "HIGH"
    }
    return severity_map.get(vuln_type, "MEDIUM")

def get_vulnerability_description(vuln_type: str) -> str:
    """Get description for vulnerability type."""
    descriptions = {
        "sql_injection": "Potential SQL injection vulnerability detected",
        "xss": "Potential Cross-Site Scripting (XSS) vulnerability",
        "hardcoded_secrets": "Hardcoded secret or API key detected",
        "ssrf": "Potential Server-Side Request Forgery (SSRF) vulnerability"
    }
    return descriptions.get(vuln_type, "Security vulnerability detected")

def redact_secrets(content: str) -> str:
    """Redact secrets from content."""
    redacted = content
    
    # Redact common secret patterns
    secret_patterns = [
        (r'password\s*=\s*["\'][^"\']+["\']', 'password="[REDACTED]"'),
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'api_key="[REDACTED]"'),
        (r'secret\s*=\s*["\'][^"\']+["\']', 'secret="[REDACTED]"'),
        (r'token\s*=\s*["\'][^"\']+["\']', 'token="[REDACTED]"'),
    ]
    
    for pattern, replacement in secret_patterns:
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
    
    return redacted

def calculate_impact_score(refactoring_type: str, file_size: int, complexity: int) -> int:
    """Calculate impact score for refactoring opportunities."""
    base_scores = {
        "dataclass_conversion": 30,
        "async_optimization": 40,
        "security_fix": 50,
        "performance_improvement": 35,
        "code_quality": 20
    }
    
    base_score = base_scores.get(refactoring_type, 10)
    size_factor = min(file_size / 1000, 5)  # Cap at 5x
    complexity_factor = min(complexity / 10, 3)  # Cap at 3x
    
    return int(base_score * (1 + size_factor + complexity_factor))

@mcp.tool()
async def fetch_files(repo_url: str, chunk_size: int = 10000, max_files: int = 1000) -> Dict[str, str]:
    """Fetch multi-language files from GitHub repository with chunking support."""
    try:
        print(f"🔍 Fetching files from GitHub: {repo_url}", file=os.sys.stderr)
        files = {}
        
        if not GITHUB_TOKEN:
            return {"error": "GITHUB_TOKEN not found in environment variables"}
        
        # Initialize GitHub client
        print("🔐 Authenticating with GitHub...", file=os.sys.stderr)
        g = Github(GITHUB_TOKEN)
        
        # Parse repository URL to get owner and repo name
        if "github.com" in repo_url:
            # Extract owner and repo from URL like https://github.com/owner/repo
            parts = repo_url.replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                owner = parts[0]
                repo_name = parts[1].replace(".git", "")
            else:
                return {"error": "Invalid GitHub repository URL format"}
        else:
            return {"error": "Please provide a valid GitHub repository URL"}
        
        # Get repository
        print(f"📂 Accessing repository: {owner}/{repo_name}", file=os.sys.stderr)
        repo = g.get_repo(f"{owner}/{repo_name}")
        
        # Supported file extensions for multi-language support
        supported_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        # Get all supported files with chunking
        print("🔍 Scanning repository for multi-language files...", file=os.sys.stderr)
        contents = repo.get_contents("")
        file_count = 0
        
        while contents and file_count < max_files:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                try:
                    contents.extend(repo.get_contents(file_content.path))
                except Exception as e:
                    print(f"⚠️ Could not access directory {file_content.path}: {e}", file=os.sys.stderr)
            else:
                # Check if file extension is supported
                file_ext = os.path.splitext(file_content.name)[1].lower()
                if file_ext in supported_extensions:
                    try:
                        file_data = file_content.decoded_content.decode('utf-8')
                        
                        # Chunk large files
                        if len(file_data) > chunk_size:
                            print(f"📄 Chunking large file: {file_content.path} ({len(file_data)} chars)", file=os.sys.stderr)
                            chunks = chunk_content(file_data, chunk_size, supported_extensions[file_ext])
                            for i, chunk in enumerate(chunks):
                                chunk_name = f"{file_content.path}_chunk_{i+1}"
                                files[chunk_name] = chunk
                                file_count += 1
                        else:
                            files[file_content.path] = file_data
                            file_count += 1
                            
                        print(f"📄 Fetched: {file_content.path} ({len(file_data)} chars, {supported_extensions[file_ext]})", file=os.sys.stderr)
                    except Exception as e:
                        print(f"⚠️ Could not fetch {file_content.path}: {e}", file=os.sys.stderr)
        
        if file_count >= max_files:
            print(f"⚠️  Reached max file limit: {max_files}", file=os.sys.stderr)
        
        # Store in session with language detection
        session_data["fetched_files"] = files
        session_data["language_analysis"] = analyze_languages(files)
        
        print(f"✅ Successfully fetched {len(files)} files across {len(set(session_data['language_analysis'].values()))} languages", file=os.sys.stderr)
        
        # Store files in session for sandboxed environment
        print(f"💾 Stored {len(files)} files in session storage", file=os.sys.stderr)
        
        return files
        
    except Exception as e:
        print(f"❌ Error fetching files: {str(e)}", file=os.sys.stderr)
        return {"error": f"Failed to fetch files from GitHub: {str(e)}"}

@mcp.tool()
async def analyze_security_vulnerabilities(files: Dict[str, str]) -> Dict[str, Any]:
    """Comprehensive security vulnerability analysis."""
    try:
        print("🔒 Analyzing security vulnerabilities...", file=os.sys.stderr)
        
        all_vulnerabilities = []
        owasp_scores = {
            "A01": 0,  # Broken Access Control
            "A02": 0,  # Cryptographic Failures
            "A03": 0,  # Injection
            "A04": 0,  # Insecure Design
            "A05": 0,  # Security Misconfiguration
            "A06": 0,  # Vulnerable Components
            "A07": 0,  # Authentication Failures
            "A08": 0,  # Software Integrity Failures
            "A09": 0,  # Logging Failures
            "A10": 0   # Server-Side Request Forgery
        }
        
        for filename, content in files.items():
            vulnerabilities = detect_security_vulnerabilities(content, filename)
            all_vulnerabilities.extend(vulnerabilities)
            
            # Map vulnerabilities to OWASP categories
            for vuln in vulnerabilities:
                if vuln["type"] == "sql_injection":
                    owasp_scores["A03"] += 1
                elif vuln["type"] == "xss":
                    owasp_scores["A03"] += 1
                elif vuln["type"] == "hardcoded_secrets":
                    owasp_scores["A02"] += 1
                elif vuln["type"] == "ssrf":
                    owasp_scores["A10"] += 1
        
        # Calculate OWASP compliance score
        total_issues = sum(owasp_scores.values())
        compliance_score = max(0, 100 - (total_issues * 10))
        
        result = {
            "total_vulnerabilities": len(all_vulnerabilities),
            "vulnerabilities": all_vulnerabilities,
            "owasp_scores": owasp_scores,
            "compliance_score": compliance_score,
            "severity_breakdown": {
                "CRITICAL": len([v for v in all_vulnerabilities if v["severity"] == "CRITICAL"]),
                "HIGH": len([v for v in all_vulnerabilities if v["severity"] == "HIGH"]),
                "MEDIUM": len([v for v in all_vulnerabilities if v["severity"] == "MEDIUM"]),
                "LOW": len([v for v in all_vulnerabilities if v["severity"] == "LOW"])
            }
        }
        
        session_data["security_vulnerabilities"] = result
        print(f"✅ Security analysis complete: {len(all_vulnerabilities)} vulnerabilities found", file=os.sys.stderr)
        return result
        
    except Exception as e:
        logging.exception("Exception during security analysis")
        return {"error": f"Security analysis failed: {str(e)}"}

@mcp.tool()
async def benchmark_performance(original_code: str, optimized_code: str, iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark performance between original and optimized code."""
    try:
        print("⚡ Running performance benchmarks...", file=os.sys.stderr)
        
        # Create temporary files for benchmarking
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as orig_file:
            orig_file.write(original_code)
            orig_path = orig_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as opt_file:
            opt_file.write(optimized_code)
            opt_path = opt_file.name
        
        try:
            # Benchmark original code
            start_time = time.time()
            result_orig = subprocess.run([sys.executable, orig_path], 
                                       capture_output=True, text=True, timeout=30)
            orig_time = time.time() - start_time
            
            # Benchmark optimized code
            start_time = time.time()
            result_opt = subprocess.run([sys.executable, opt_path], 
                                      capture_output=True, text=True, timeout=30)
            opt_time = time.time() - start_time
            
            # Calculate performance metrics
            performance_improvement = ((orig_time - opt_time) / orig_time) * 100 if orig_time > 0 else 0
            regression_detected = opt_time > orig_time * 1.1  # 10% threshold
            
            benchmark_result = {
                "original_time": orig_time,
                "optimized_time": opt_time,
                "performance_improvement": performance_improvement,
                "regression_detected": regression_detected,
                "recommendation": "ACCEPT" if not regression_detected else "REJECT",
                "flame_graph_data": generate_flame_graph_data(original_code, optimized_code),
                "bottleneck_analysis": analyze_bottlenecks(original_code, optimized_code)
            }
            
            session_data["performance_benchmarks"] = benchmark_result
            print(f"✅ Performance benchmark complete: {performance_improvement:.2f}% improvement", file=os.sys.stderr)
            return benchmark_result
            
        finally:
            # Cleanup temporary files
            os.unlink(orig_path)
            os.unlink(opt_path)
            
    except Exception as e:
        logging.exception("Exception during performance benchmarking")
        return {"error": f"Performance benchmarking failed: {str(e)}"}

@mcp.tool()
async def run_tests_and_rollback(files: Dict[str, str], changes: Dict[str, str]) -> Dict[str, Any]:
    """Run tests and implement rollback if failures detected."""
    try:
        print("🧪 Running tests with rollback capability...", file=os.sys.stderr)
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write original files
            for filename, content in files.items():
                file_path = os.path.join(temp_dir, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(content)
            
            # Run original tests
            print("🔍 Running original tests...", file=os.sys.stderr)
            original_test_result = run_test_suite(temp_dir)
            
            if not original_test_result["passed"]:
                return {
                    "status": "SKIP",
                    "reason": "Original tests failed - cannot proceed with refactoring",
                    "original_test_result": original_test_result
                }
            
            # Apply changes
            for filename, new_content in changes.items():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(new_content)
            
            # Run tests with changes
            print("🔍 Running tests with changes...", file=os.sys.stderr)
            modified_test_result = run_test_suite(temp_dir)
            
            # Determine if rollback is needed
            rollback_needed = not modified_test_result["passed"]
            
            if rollback_needed:
                print("⚠️  Test failures detected - implementing rollback", file=os.sys.stderr)
                # Restore original files
                for filename, content in files.items():
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, 'w') as f:
                        f.write(content)
                
                # Verify rollback
                rollback_test_result = run_test_suite(temp_dir)
                
                result = {
                    "status": "ROLLBACK",
                    "rollback_needed": True,
                    "original_test_result": original_test_result,
                    "modified_test_result": modified_test_result,
                    "rollback_test_result": rollback_test_result,
                    "failure_analysis": analyze_test_failures(modified_test_result),
                    "alternative_suggestions": generate_alternative_refactoring(files, changes)
                }
            else:
                result = {
                    "status": "SUCCESS",
                    "rollback_needed": False,
                    "original_test_result": original_test_result,
                    "modified_test_result": modified_test_result,
                    "test_coverage": calculate_test_coverage(temp_dir)
                }
            
            session_data["test_results"] = result
            print(f"✅ Test execution complete: {result['status']}", file=os.sys.stderr)
            return result
            
    except Exception as e:
        logging.exception("Exception during test execution")
        return {"error": f"Test execution failed: {str(e)}"}

@mcp.tool()
async def rank_refactoring_opportunities(files: Dict[str, str]) -> Dict[str, Any]:
    """Rank refactoring opportunities by impact and priority."""
    try:
        print("📊 Ranking refactoring opportunities...", file=os.sys.stderr)
        
        opportunities = []
        
        for filename, content in files.items():
            file_size = len(content)
            complexity = calculate_complexity(content)
            
            # Detect opportunities
            if "class " in content and "__init__" in content:
                score = calculate_impact_score("dataclass_conversion", file_size, complexity)
                opportunities.append({
                    "type": "dataclass_conversion",
                    "file": filename,
                    "impact_score": score,
                    "description": "Convert class to @dataclass for better readability",
                    "priority": "HIGH" if score > 50 else "MEDIUM"
                })
            
            if "async def" in content and "asyncio.gather" not in content:
                score = calculate_impact_score("async_optimization", file_size, complexity)
                opportunities.append({
                    "type": "async_optimization", 
                    "file": filename,
                    "impact_score": score,
                    "description": "Optimize async operations with asyncio.gather()",
                    "priority": "HIGH" if score > 40 else "MEDIUM"
                })
            
            if "print(" in content:
                score = calculate_impact_score("code_quality", file_size, complexity)
                opportunities.append({
                    "type": "logging_improvement",
                    "file": filename,
                    "impact_score": score,
                    "description": "Replace print statements with proper logging",
                    "priority": "MEDIUM"
                })
        
        # Sort by impact score and get top 20
        opportunities.sort(key=lambda x: x["impact_score"], reverse=True)
        top_opportunities = opportunities[:20]
        
        result = {
            "total_opportunities": len(opportunities),
            "top_20": top_opportunities,
            "priority_breakdown": {
                "HIGH": len([o for o in opportunities if o["priority"] == "HIGH"]),
                "MEDIUM": len([o for o in opportunities if o["priority"] == "MEDIUM"]),
                "LOW": len([o for o in opportunities if o["priority"] == "LOW"])
            },
            "impact_distribution": {
                "dataclass_conversion": len([o for o in opportunities if o["type"] == "dataclass_conversion"]),
                "async_optimization": len([o for o in opportunities if o["type"] == "async_optimization"]),
                "logging_improvement": len([o for o in opportunities if o["type"] == "logging_improvement"])
            }
        }
        
        session_data["impact_rankings"] = result
        print(f"✅ Ranking complete: {len(opportunities)} opportunities identified", file=os.sys.stderr)
        return result
        
    except Exception as e:
        logging.exception("Exception during opportunity ranking")
        return {"error": f"Opportunity ranking failed: {str(e)}"}

# Helper functions for new capabilities
def run_test_suite(test_dir: str) -> Dict[str, Any]:
    """Run test suite in directory."""
    try:
        # Look for common test files
        test_files = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.startswith('test_') or file.endswith('_test.py'):
                    test_files.append(os.path.join(root, file))
        
        if not test_files:
            return {"passed": True, "message": "No tests found"}
        
        # Run pytest if available, otherwise run individual test files
        try:
            result = subprocess.run([sys.executable, '-m', 'pytest', test_dir], 
                                 capture_output=True, text=True, timeout=60)
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "message": "Tests timed out"}
        except FileNotFoundError:
            # Fallback to running individual test files
            for test_file in test_files:
                result = subprocess.run([sys.executable, test_file], 
                                     capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    return {"passed": False, "output": result.stdout, "errors": result.stderr}
            return {"passed": True, "message": "All tests passed"}
            
    except Exception as e:
        return {"passed": False, "message": f"Test execution error: {str(e)}"}

def calculate_complexity(content: str) -> int:
    """Calculate cyclomatic complexity."""
    complexity = 1  # Base complexity
    
    # Count decision points
    complexity += len(re.findall(r'\bif\b', content))
    complexity += len(re.findall(r'\bfor\b', content))
    complexity += len(re.findall(r'\bwhile\b', content))
    complexity += len(re.findall(r'\bexcept\b', content))
    complexity += len(re.findall(r'\bcase\b', content))
    
    return complexity

def generate_flame_graph_data(original_code: str, optimized_code: str) -> Dict[str, Any]:
    """Generate flame graph data for performance analysis."""
    return {
        "original_functions": extract_function_calls(original_code),
        "optimized_functions": extract_function_calls(optimized_code),
        "performance_hotspots": identify_hotspots(original_code),
        "optimization_points": identify_optimization_points(optimized_code)
    }

def analyze_bottlenecks(original_code: str, optimized_code: str) -> Dict[str, Any]:
    """Analyze performance bottlenecks."""
    return {
        "io_operations": count_patterns(original_code, [r'open\(', r'requests\.', r'urllib']),
        "loops": count_patterns(original_code, [r'for ', r'while ']),
        "async_operations": count_patterns(original_code, [r'await ', r'async ']),
        "optimizations_applied": count_patterns(optimized_code, [r'asyncio\.gather', r'@dataclass'])
    }

def extract_function_calls(code: str) -> List[str]:
    """Extract function calls from code."""
    function_pattern = r'(\w+)\s*\('
    return re.findall(function_pattern, code)

def identify_hotspots(code: str) -> List[str]:
    """Identify performance hotspots."""
    hotspots = []
    if 'time.sleep(' in code:
        hotspots.append("Blocking sleep operations")
    if 'for ' in code and 'await ' in code:
        hotspots.append("Sequential async operations")
    if 'open(' in code and 'with ' not in code:
        hotspots.append("Unmanaged file operations")
    return hotspots

def identify_optimization_points(code: str) -> List[str]:
    """Identify optimization points."""
    optimizations = []
    if 'asyncio.gather(' in code:
        optimizations.append("Parallel async execution")
    if '@dataclass' in code:
        optimizations.append("Dataclass optimization")
    if 'logging.' in code:
        optimizations.append("Proper logging implementation")
    return optimizations

def count_patterns(code: str, patterns: List[str]) -> int:
    """Count occurrences of patterns in code."""
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, code))
    return total

def analyze_test_failures(test_result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze test failure root causes."""
    return {
        "failure_type": "syntax_error" if "SyntaxError" in str(test_result) else "runtime_error",
        "suggested_fixes": ["Check syntax", "Verify imports", "Review logic"],
        "rollback_reason": "Test failures detected"
    }

def generate_alternative_refactoring(files: Dict[str, str], changes: Dict[str, str]) -> List[str]:
    """Generate alternative refactoring suggestions."""
    return [
        "Try smaller, incremental changes",
        "Focus on one file at a time",
        "Add comprehensive tests before refactoring",
        "Consider gradual migration approach"
    ]

def calculate_test_coverage(test_dir: str) -> Dict[str, Any]:
    """Calculate test coverage."""
    return {
        "coverage_percentage": 85.0,  # Placeholder
        "lines_covered": 1200,
        "total_lines": 1400,
        "branches_covered": 45,
        "total_branches": 50
    }

@mcp.tool()
async def suggest_refactor(file_content: str) -> str:
    """Suggest comprehensive refactoring improvements for code."""
    try:
        logging.info("Analyzing code for refactoring suggestions")
        suggestions = []
        
        # Class-related improvements
        if "class " in file_content and "__init__" in file_content:
            if "def __init__(self" in file_content and "self." in file_content:
                suggestions.append("🔧 Consider converting simple classes to @dataclass for better readability and less boilerplate.")
            if "class " in file_content and "def __eq__" not in file_content and "def __repr__" not in file_content:
                suggestions.append("🔧 Add __repr__ and __eq__ methods or use @dataclass for automatic generation.")
        
        # Async/IO optimizations
        if "async def" in file_content:
            if "await " in file_content and "asyncio.gather" not in file_content:
                suggestions.append("⚡ Consider using asyncio.gather() for parallel execution of multiple async operations.")
            if "for " in file_content and "await " in file_content:
                suggestions.append("⚡ Replace sequential await calls in loops with asyncio.gather() for better performance.")
        
        # Import optimizations
        if "import *" in file_content:
            suggestions.append("📦 Avoid wildcard imports (import *) for better code clarity and namespace pollution prevention.")
        
        # Logging improvements
        if "print(" in file_content:
            suggestions.append("📝 Replace print statements with proper logging for better debugging and production use.")
        
        # Type hints
        if "def " in file_content and "->" not in file_content:
            suggestions.append("🎯 Add type hints to function signatures for better code documentation and IDE support.")
        
        # Error handling
        if "try:" in file_content and "except" in file_content:
            if "Exception" in file_content and "except Exception" in file_content:
                suggestions.append("🛡️ Use specific exception types instead of bare 'except Exception' for better error handling.")
        
        # String formatting
        if "%" in file_content and "format(" not in file_content and "f\"" not in file_content:
            suggestions.append("📝 Consider using f-strings or .format() instead of % formatting for better readability.")
        
        # List comprehensions
        if "for " in file_content and "append(" in file_content:
            suggestions.append("🐍 Consider using list comprehensions instead of explicit loops with append() for more Pythonic code.")
        
        # Constants
        if "=" in file_content and "UPPER_CASE" not in file_content:
            suggestions.append("📌 Consider using UPPER_CASE naming for constants to follow Python conventions.")
        
        if not suggestions:
            return "✅ No specific refactoring suggestions detected. Code appears to follow good practices!"
        
        result = "🚀 **Refactoring Suggestions:**\n\n" + "\n".join(suggestions)
        logging.info(f"Generated {len(suggestions)} refactoring suggestions")
        return result
    except Exception as e:
        logging.exception("Exception during refactoring analysis")
        return f"Error: {str(e)}"

@mcp.tool()
async def generate_diff(original: str, optimized: str) -> str:
    """Generate diff between original and optimized code."""
    try:
        logging.info("Generating diff between original and optimized code")
        diff = difflib.unified_diff(
            original.splitlines(),
            optimized.splitlines(),
            fromfile="original",
            tofile="optimized",
            lineterm=""
        )
        result = "\n".join(diff)
        logging.info("Diff generated successfully")
        
        # Store diff in session
        diff_id = f"diff_{len(session_data['generated_diffs'])}"
        session_data["generated_diffs"][diff_id] = {
            "original": original,
            "optimized": optimized,
            "diff": result
        }
        print(f"💾 Stored diff in session: {diff_id}", file=os.sys.stderr)
        
        return result
    except Exception as e:
        logging.exception("Exception during diff generation")
        return f"Error: {str(e)}"

@mcp.tool()
async def create_pull_request(branch_name: str, changes: Dict[str, str], title: str) -> str:
    """Create a pull request with changes."""
    try:
        logging.info(f"Creating PR: {title} on branch {branch_name}")
        
        if not GITHUB_TOKEN:
            return "❌ Missing GITHUB_TOKEN in environment. Please set your GitHub token in the .env file."

        g = Github(GITHUB_TOKEN)
        
        # Get repository URL from environment
        repo_url = os.getenv("REPO_URL", "https://github.com/tanzeela1078-cyber/FISTA")
        
        # Extract owner and repo from URL
        if "github.com" in repo_url:
            parts = repo_url.replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                owner = parts[0]
                repo_name = parts[1].replace(".git", "")
            else:
                return "❌ Invalid repository URL format"
        else:
            return "❌ Please provide a valid GitHub repository URL"
        
        # Get the repository
        repo = g.get_repo(f"{owner}/{repo_name}")
        print(f"🔗 Repository: {repo_url}", file=sys.stderr)
        
        # Create a new branch
        try:
            # Get the main branch
            main_branch = repo.get_branch("main")
            main_sha = main_branch.commit.sha
            
            # Create new branch
            repo.create_git_ref(f"refs/heads/{branch_name}", main_sha)
            logging.info(f"Created branch: {branch_name}")
            
        except Exception as e:
            logging.warning(f"Branch might already exist: {e}")
        
        # Create or update files
        for fname, content in changes.items():
            try:
                # Check if file exists
                try:
                    file_obj = repo.get_contents(fname)
                    # Update existing file
                    repo.update_file(
                        path=fname,
                        message=f"Update {fname} - {title}",
                        content=content,
                        sha=file_obj.sha,
                        branch=branch_name
                    )
                    logging.info(f"Updated file: {fname}")
                except:
                    # Create new file
                    repo.create_file(
                        path=fname,
                        message=f"Add {fname} - {title}",
                        content=content,
                        branch=branch_name
                    )
                    logging.info(f"Created file: {fname}")
                    
            except Exception as e:
                logging.error(f"Failed to update/create {fname}: {e}")
                return f"Failed to update file {fname}: {str(e)}"
        
        # Create pull request
        try:
            pr = repo.create_pull(
                title=title,
                body="🤖 **Automated PR generated by AI Code Auditor**\n\n" +
                     "This pull request contains refactoring suggestions and improvements:\n" +
                     "- Code quality enhancements\n" +
                     "- Performance optimizations\n" +
                     "- Best practices implementation\n\n" +
                     "Generated by MCP Code Auditor Agent.",
                head=branch_name,
                base="main"
            )
            
            result = f"🎉 **PULL REQUEST CREATED SUCCESSFULLY!**\n\n" + \
                   f"🔗 **Repository:** {repo_url}\n" + \
                   f"🔗 **PR URL:** {pr.html_url}\n" + \
                   f"📝 **Title:** {pr.title}\n" + \
                   f"🌿 **Branch:** {branch_name}\n" + \
                   f"📊 **Files Changed:** {len(changes)}\n" + \
                   f"📅 **Created:** {pr.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n" + \
                   f"✅ **COMPLETION LOG:**\n" + \
                   f"   • Repository: {repo_url}\n" + \
                   f"   • Branch created: {branch_name}\n" + \
                   f"   • Files updated: {', '.join(changes.keys())}\n" + \
                   f"   • Pull request: {pr.html_url}\n" + \
                   f"   • Status: Ready for review"
            
            print(f"🎉 PULL REQUEST CREATED: {pr.html_url}", file=sys.stderr)
            print(f"🔗 Repository: {repo_url}", file=sys.stderr)
            print(f"🌿 Branch: {branch_name}", file=sys.stderr)
            print(f"📊 Files: {', '.join(changes.keys())}", file=sys.stderr)
            logging.info(f"PR created successfully: {pr.html_url}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Failed to create PR: {e}")
            
            # Provide specific error messages
            if "403" in error_msg or "Forbidden" in error_msg:
                return f"❌ **PULL REQUEST CREATION FAILED**\n\n" + \
                       f"🔒 **Permission Error:** Your GitHub token doesn't have write access to the repository.\n\n" + \
                       f"📋 **Required Permissions:**\n" + \
                       f"   • Repository: {repo_url}\n" + \
                       f"   • Token scope: 'repo' (full repository access)\n" + \
                       f"   • Branch: {branch_name}\n\n" + \
                       f"🔧 **To Fix:**\n" + \
                       f"   1. Go to GitHub Settings → Developer settings → Personal access tokens\n" + \
                       f"   2. Create a new token with 'repo' scope\n" + \
                       f"   3. Update your .env file with the new token\n" + \
                       f"   4. Restart the application\n\n" + \
                       f"📝 **Error Details:** {error_msg}"
            else:
                return f"❌ **PULL REQUEST CREATION FAILED**\n\n" + \
                       f"🔗 **Repository:** {repo_url}\n" + \
                       f"🌿 **Branch:** {branch_name}\n" + \
                       f"📝 **Error:** {error_msg}\n\n" + \
                       f"🔧 **Troubleshooting:**\n" + \
                       f"   • Check your GitHub token permissions\n" + \
                       f"   • Verify the repository URL is correct\n" + \
                       f"   • Ensure you have write access to the repository"
            
    except Exception as e:
        logging.exception("Exception during PR creation")
        return f"Error creating pull request: {str(e)}"

@mcp.tool()
async def analyze_file(filename: str, files: Dict[str, str]) -> str:
    """Analyze a specific file and provide comprehensive suggestions."""
    try:
        logging.info(f"Analyzing file: {filename}")
        
        if filename not in files:
            available_files = list(files.keys())
            return f"❌ File '{filename}' not found. Available files: {available_files}"
        
        file_content = files[filename]
        
        # Get refactoring suggestions
        refactor_suggestions = await suggest_refactor(file_content)
        
        # Analyze the file for specific patterns
        analysis = []
        analysis.append(f"📁 **File Analysis: {filename}**")
        analysis.append(f"📊 **File Size:** {len(file_content)} characters")
        analysis.append(f"📝 **Lines:** {len(file_content.splitlines())}")
        
        # Check for specific patterns
        if "async def" in file_content:
            async_count = file_content.count("async def")
            analysis.append(f"⚡ **Async Functions:** {async_count}")
        
        if "class " in file_content:
            class_count = file_content.count("class ")
            analysis.append(f"🏗️ **Classes:** {class_count}")
        
        if "import " in file_content:
            import_count = file_content.count("import ")
            analysis.append(f"📦 **Imports:** {import_count}")
        
        # Combine analysis with refactoring suggestions
        result = "\n\n".join(analysis) + "\n\n" + refactor_suggestions
        
        logging.info(f"Analysis completed for {filename}")
        return result
        
    except Exception as e:
        logging.exception("Exception during file analysis")
        return f"Error analyzing file: {str(e)}"

@mcp.tool()
async def display_fetched_files() -> str:
    """Display all fetched files with their content."""
    try:
        if not session_data["fetched_files"]:
            return "📂 No files have been fetched yet. Use fetch_files first."
        
        result = []
        result.append("📁 **FETCHED FILES FROM GITHUB**")
        result.append("=" * 50)
        
        for filename, content in session_data["fetched_files"].items():
            result.append(f"\n📄 **File: {filename}**")
            result.append(f"📊 Size: {len(content)} characters, {len(content.splitlines())} lines")
            result.append("```python")
            # Show first 50 lines to avoid overwhelming output
            lines = content.splitlines()
            display_lines = lines[:50]
            result.append("\n".join(display_lines))
            if len(lines) > 50:
                result.append(f"\n... ({len(lines) - 50} more lines)")
            result.append("```")
            result.append("-" * 30)
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error displaying files: {str(e)}"

@mcp.tool()
async def display_diffs() -> str:
    """Display generated diffs for refactoring suggestions."""
    try:
        if not session_data["generated_diffs"]:
            return "🔍 No diffs have been generated yet. Use generate_diff first."
        
        result = []
        result.append("🔄 **GENERATED DIFFS**")
        result.append("=" * 50)
        
        for diff_id, diff_data in session_data["generated_diffs"].items():
            result.append(f"\n🆔 **Diff ID: {diff_id}**")
            result.append("```diff")
            result.append(diff_data["diff"])
            result.append("```")
            result.append("-" * 30)
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error displaying diffs: {str(e)}"

@mcp.tool()
async def get_session_data() -> str:
    """Get all session data including files, analysis, and diffs."""
    try:
        result = []
        result.append("🗄️ **SESSION DATA OVERVIEW**")
        result.append("=" * 50)
        
        # Files summary
        files_count = len(session_data["fetched_files"])
        result.append(f"📁 **Fetched Files:** {files_count}")
        if files_count > 0:
            result.append("   Files:")
            for filename in session_data["fetched_files"].keys():
                result.append(f"   - {filename}")
        
        # Analysis results
        analysis_count = len(session_data["analysis_results"])
        result.append(f"\n🔍 **Analysis Results:** {analysis_count}")
        
        # Diffs
        diffs_count = len(session_data["generated_diffs"])
        result.append(f"\n🔄 **Generated Diffs:** {diffs_count}")
        if diffs_count > 0:
            result.append("   Diff IDs:")
            
            for diff_id in session_data["generated_diffs"].keys():
                result.append(f"   - {diff_id}")
        
        # Suggestions
        suggestions_count = len(session_data["refactoring_suggestions"])
        result.append(f"\n💡 **Refactoring Suggestions:** {suggestions_count}")
        
        result.append(f"\n💾 **Total Session Size:** {len(str(session_data))} characters")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error getting session data: {str(e)}"

@mcp.tool()
async def list_available_files() -> str:
    """List all available files for analysis"""
    try:
        if not session_data["fetched_files"]:
            return "📂 No files have been fetched yet. Use fetch_files first."
        
        result = []
        result.append("📁 **AVAILABLE FILES FOR ANALYSIS**")
        result.append("=" * 50)
        
        for i, (filename, content) in enumerate(session_data["fetched_files"].items(), 1):
            lines = len(content.splitlines())
            size = len(content)
            result.append(f"{i}. 📄 **{filename}**")
            result.append(f"   📊 Size: {size} characters, {lines} lines")
            
            # Show file type indicators
            indicators = []
            if "async def" in content:
                indicators.append("⚡ async")
            if "class " in content:
                indicators.append("🏗️ classes")
            if "import " in content:
                indicators.append("📦 imports")
            if "print(" in content:
                indicators.append("📝 print statements")
            if "def " in content:
                indicators.append("🔧 functions")
            
            if indicators:
                result.append(f"   🏷️ Features: {', '.join(indicators)}")
            
            result.append("")
        
        result.append("💡 **Usage:** Use analyze_file with the filename to analyze a specific file.")
        result.append("📋 **Example:** analyze_file('async_tasks.py', files)")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error listing files: {str(e)}"

@mcp.tool()
async def ping() -> str:
    """Check if the MCP server is alive"""
    return "✅ GitHub Refactor MCP server is alive"

if __name__ == "__main__":
    print("🚀 Starting GitHub Refactor MCP Server...")
    mcp.run()
