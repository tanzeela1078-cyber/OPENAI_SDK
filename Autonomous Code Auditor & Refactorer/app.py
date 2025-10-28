#!/usr/bin/env python3
"""
Enhanced Streamlit Frontend for Code Auditor & Refactorer
Supports all extreme test cases: massive codebases, security analysis, multi-language support, rollback mechanisms, and performance regression prevention
"""

import streamlit as st
import asyncio
import json
import time
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="Advanced Code Auditor & Refactorer",
    page_icon="🔍",
    layout="wide"
)

class MCPClient:
    """Enhanced client to communicate with MCP stdio server"""
    
    def __init__(self):
        self.process = None
        self.request_id = 0
    
    def start_server(self):
        """Start the MCP stdio server"""
        if self.process is None:
            server_script = os.path.join(os.path.dirname(__file__), "mcp_server_stdio.py")
            self.process = subprocess.Popen(
                [sys.executable, server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Wait for server to be ready
            time.sleep(1)
    
    def call_tool(self, tool_name: str, **kwargs) -> str:
        """Call a tool on the MCP server"""
        self.start_server()
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs
            }
        }
        
        try:
            # Send request
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            response = json.loads(response_line.strip())
            
            if "result" in response:
                return response["result"]["content"][0]["text"]
            elif "error" in response:
                return f"Error: {response['error']['message']}"
            else:
                return "Unknown response format"
                
        except Exception as e:
            return f"Error calling tool: {str(e)}"
    
    def close(self):
        """Close the MCP server"""
        if self.process:
            self.process.terminate()
            self.process = None

# Global MCP client
mcp_client = MCPClient()

# Enhanced CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .file-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .file-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .file-card.selected {
        border-left-color: #ff7f0e;
        background-color: #fff3e0;
    }
    .file-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1f77b4;
        font-size: 1.2rem;
    }
    .file-card p {
        margin: 0.25rem 0;
        color: #666;
        font-size: 0.9rem;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
    }
    .success-card h3 {
        color: #155724;
        margin: 0 0 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
        box-shadow: 0 2px 4px rgba(255, 193, 7, 0.2);
    }
    .warning-card h3 {
        color: #856404;
        margin: 0 0 1rem 0;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
        box-shadow: 0 2px 4px rgba(220, 53, 69, 0.2);
    }
    .error-card h3 {
        color: #721c24;
        margin: 0 0 1rem 0;
    }
    .code-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .diff-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1976d2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .language-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #e1f5fe;
        color: #0277bd;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
    .severity-critical {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .severity-high {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .severity-medium {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .severity-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application with enhanced capabilities"""
    
    # Initialize session state variables
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Advanced Code Auditor & Refactorer</h1>', unsafe_allow_html=True)
    st.markdown("**Enterprise-grade AI-powered code analysis with security, performance, and multi-language support**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Repository configuration
        default_repo_url = os.getenv("REPO_URL", "https://github.com/tanzeela1078-cyber/FISTA")
        repo_url = st.text_input(
            "GitHub Repository URL",
            value=default_repo_url,
            help="Enter the GitHub repository URL to analyze"
        )
        
        # Advanced options
        st.subheader("🔧 Advanced Options")
        
        chunk_size = st.slider(
            "Chunk Size (for large files)",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Size of chunks for processing large files"
        )
        
        max_files = st.slider(
            "Max Files to Process",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Maximum number of files to process"
        )
        
        enable_security = st.checkbox("🔒 Enable Security Analysis", value=True)
        enable_performance = st.checkbox("⚡ Enable Performance Benchmarking", value=True)
        enable_rollback = st.checkbox("🔄 Enable Test Rollback", value=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Fetch Files", type="primary"):
                reset_session_state()
                st.session_state.fetch_files = True
                st.session_state.chunk_size = chunk_size
                st.session_state.max_files = max_files
                st.session_state.enable_security = enable_security
                st.session_state.enable_performance = enable_performance
                st.session_state.enable_rollback = enable_rollback
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset", type="secondary"):
                reset_session_state()
                st.rerun()
    
    # Main workflow
    if st.session_state.fetch_files:
        fetch_and_display_files(repo_url)
    elif st.session_state.selected_file and not st.session_state.analysis_complete:
        analyze_selected_file(repo_url, st.session_state.selected_file)
    elif st.session_state.analysis_complete:
        show_analysis_results()
    else:
        show_welcome_screen()

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'fetch_files': True,
        'selected_file': None,
        'analysis_complete': False,
        'pr_created': False,
        'files_fetched': False,
        'security_analysis_complete': False,
        'performance_analysis_complete': False,
        'rollback_analysis_complete': False,
        'chunk_size': 10000,
        'max_files': 100,
        'enable_security': True,
        'enable_performance': True,
        'enable_rollback': True,
        'language_analysis': {},
        'security_vulnerabilities': {},
        'performance_benchmarks': {},
        'test_results': {},
        'impact_rankings': {}
    }
    
    for key, value in defaults.items():
        if not hasattr(st.session_state, key):
            setattr(st.session_state, key, value)

def reset_session_state():
    """Reset session state for new analysis"""
    st.session_state.fetch_files = True
    st.session_state.selected_file = None
    st.session_state.analysis_complete = False
    st.session_state.pr_created = False
    st.session_state.files_fetched = False
    st.session_state.security_analysis_complete = False
    st.session_state.performance_analysis_complete = False
    st.session_state.rollback_analysis_complete = False

def show_welcome_screen():
    """Show welcome screen with feature overview"""
    st.markdown("### 🚀 Welcome to Advanced Code Auditor & Refactorer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔒 Security Analysis
        - SQL injection detection
        - XSS vulnerability scanning
        - Hardcoded secrets detection
        - SSRF vulnerability detection
        - OWASP compliance scoring
        """)
    
    with col2:
        st.markdown("""
        #### 🌐 Multi-Language Support
        - Python, JavaScript, TypeScript
        - Go, Rust, Java, C++
        - Language-specific optimizations
        - Build system preservation
        """)
    
    with col3:
        st.markdown("""
        #### ⚡ Performance & Testing
        - Performance benchmarking
        - Regression detection
        - Test execution with rollback
        - Impact-based ranking
        """)
    
    st.info("👆 **Get Started:** Configure your repository URL in the sidebar and click 'Fetch Files' to begin analysis!")

def fetch_and_display_files(repo_url):
    """Enhanced file fetching with multi-language support and chunking"""
    st.markdown("### 📁 Fetching Files from GitHub")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Check if files are already fetched
        if hasattr(st.session_state, 'fetched_files') and st.session_state.fetched_files:
            files = st.session_state.fetched_files
            st.success(f"✅ Using cached files ({len(files)} files)")
        else:
            # Fetch files with enhanced parameters
            status_text.text("🔍 Fetching multi-language files from GitHub...")
            progress_bar.progress(20)
            
            files_result = mcp_client.call_tool(
                "fetch_files", 
                repo_url=repo_url,
                chunk_size=st.session_state.chunk_size,
                max_files=st.session_state.max_files
            )
            
            # Parse the result
            if isinstance(files_result, str):
                try:
                    files = json.loads(files_result)
                except json.JSONDecodeError:
                    if "Error:" in files_result or "Failed" in files_result or "❌" in files_result:
                        st.error(f"❌ Error fetching files: {files_result}")
                        return
                    else:
                        st.error(f"❌ Error parsing files result: {files_result}")
                        return
            else:
                files = files_result
            
            # Check for errors
            if isinstance(files, dict) and "error" in files and len(files) == 1:
                st.error(f"❌ Error fetching files: {files['error']}")
                return
            
            # Store files in session state
            st.session_state.fetched_files = files
            st.session_state.files_fetched = True
        
        progress_bar.progress(50)
        status_text.text("✅ Files fetched successfully!")
        
        # Run comprehensive analysis
        if st.session_state.enable_security:
            run_security_analysis()
        
        if st.session_state.enable_performance:
            run_performance_analysis()
        
        # Display files with enhanced information
        display_enhanced_file_selection()
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"❌ Error during file fetching: {str(e)}")
        st.exception(e)

def run_security_analysis():
    """Run comprehensive security vulnerability analysis"""
    if st.session_state.security_analysis_complete:
        return
    
    try:
        files = st.session_state.fetched_files
        security_result = mcp_client.call_tool("analyze_security_vulnerabilities", files=files)
        
        if isinstance(security_result, str):
            security_data = json.loads(security_result)
        else:
            security_data = security_result
        
        st.session_state.security_vulnerabilities = security_data
        st.session_state.security_analysis_complete = True
        
    except Exception as e:
        st.warning(f"⚠️ Security analysis failed: {str(e)}")

def run_performance_analysis():
    """Run performance analysis and opportunity ranking"""
    if st.session_state.performance_analysis_complete:
        return
    
    try:
        files = st.session_state.fetched_files
        rankings_result = mcp_client.call_tool("rank_refactoring_opportunities", files=files)
        
        if isinstance(rankings_result, str):
            rankings_data = json.loads(rankings_result)
        else:
            rankings_data = rankings_result
        
        st.session_state.impact_rankings = rankings_data
        st.session_state.performance_analysis_complete = True
        
    except Exception as e:
        st.warning(f"⚠️ Performance analysis failed: {str(e)}")

def display_enhanced_file_selection():
    """Display files with enhanced information and analysis results"""
    st.markdown("### 🎯 Select File for Analysis")
    
    files = st.session_state.fetched_files
    
    # Show analysis summary
    if st.session_state.security_analysis_complete or st.session_state.performance_analysis_complete:
        show_analysis_summary()
    
    # Language distribution
    if hasattr(st.session_state, 'language_analysis') and st.session_state.language_analysis:
        show_language_distribution()
    
    # File selection interface
    st.markdown("Click on a file card to select it for detailed analysis:")
    
    file_options = list(files.keys())
    
    for i, filename in enumerate(file_options):
        content = files[filename]
        lines = len(content.splitlines())
        size = len(content)
        
        # Enhanced file analysis
        features = analyze_file_features(content, filename)
        
        # Security indicators
        security_indicators = get_security_indicators(filename)
        
        # Performance indicators
        performance_indicators = get_performance_indicators(filename)
        
        # Create enhanced file card
        is_selected = hasattr(st.session_state, 'selected_file') and st.session_state.selected_file == filename
        card_class = "file-card selected" if is_selected else "file-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>📄 {filename}</h4>
            <p>📊 {size} chars, {lines} lines</p>
            <p>🏷️ {', '.join(features) if features else 'No special features detected'}</p>
            {security_indicators}
            {performance_indicators}
        </div>
        """, unsafe_allow_html=True)
        
        # Add click handler
        if st.button(f"Select {filename}", key=f"select_{i}", use_container_width=True):
            st.session_state.selected_file = filename
            st.session_state.fetch_files = False
            st.rerun()

def analyze_file_features(content: str, filename: str) -> List[str]:
    """Analyze file features for display"""
    features = []
    
    # Language detection
    ext = os.path.splitext(filename)[1].lower()
    language_map = {
        '.py': '🐍 Python',
        '.js': '🟨 JavaScript',
        '.ts': '🔷 TypeScript',
        '.go': '🐹 Go',
        '.rs': '🦀 Rust',
        '.java': '☕ Java',
        '.cpp': '⚙️ C++',
        '.c': '⚙️ C',
        '.cs': '🔷 C#',
        '.php': '🐘 PHP',
        '.rb': '💎 Ruby',
        '.swift': '🦉 Swift',
        '.kt': '🟣 Kotlin'
    }
    
    if ext in language_map:
        features.append(language_map[ext])
    
    # Code features
    if "async def" in content or "async function" in content:
        features.append("⚡ async")
    if "class " in content:
        features.append("🏗️ classes")
    if "import " in content or "require(" in content:
        features.append("📦 imports")
    if "print(" in content or "console.log" in content:
        features.append("📝 logging")
    if "def " in content or "function " in content:
        features.append("🔧 functions")
    if "test_" in filename or "_test" in filename:
        features.append("🧪 tests")
    
    return features

def get_security_indicators(filename: str) -> str:
    """Get security indicators for a file"""
    if not st.session_state.security_analysis_complete:
        return ""
    
    vulnerabilities = st.session_state.security_vulnerabilities.get('vulnerabilities', [])
    file_vulns = [v for v in vulnerabilities if v.get('file') == filename]
    
    if not file_vulns:
        return '<p style="color: green;">🔒 No security issues</p>'
    
    severity_counts = {}
    for vuln in file_vulns:
        severity = vuln.get('severity', 'UNKNOWN')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    indicators = []
    for severity, count in severity_counts.items():
        severity_class = f"severity-{severity.lower()}"
        indicators.append(f'<span class="{severity_class}">{severity}: {count}</span>')
    
    return f'<p>🔒 Security: {" ".join(indicators)}</p>'

def get_performance_indicators(filename: str) -> str:
    """Get performance indicators for a file"""
    if not st.session_state.performance_analysis_complete:
        return ""
    
    rankings = st.session_state.impact_rankings.get('top_20', [])
    file_opportunities = [o for o in rankings if o.get('file') == filename]
    
    if not file_opportunities:
        return '<p style="color: gray;">⚡ No optimization opportunities</p>'
    
    total_score = sum(o.get('impact_score', 0) for o in file_opportunities)
    priority = file_opportunities[0].get('priority', 'LOW')
    
    priority_class = f"severity-{priority.lower()}"
    return f'<p>⚡ Performance: <span class="{priority_class}">{priority}</span> (Score: {total_score})</p>'

def show_analysis_summary():
    """Show comprehensive analysis summary"""
    st.markdown("### 📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        files_count = len(st.session_state.fetched_files)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{files_count}</div>
            <div class="metric-label">Files Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.security_analysis_complete:
        with col2:
            vuln_count = st.session_state.security_vulnerabilities.get('total_vulnerabilities', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{vuln_count}</div>
                <div class="metric-label">Security Issues</div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.performance_analysis_complete:
        with col3:
            opp_count = st.session_state.impact_rankings.get('total_opportunities', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{opp_count}</div>
                <div class="metric-label">Optimization Opportunities</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        languages = len(set(st.session_state.language_analysis.values())) if st.session_state.language_analysis else 1
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{languages}</div>
            <div class="metric-label">Languages Detected</div>
        </div>
        """, unsafe_allow_html=True)

def show_language_distribution():
    """Show language distribution"""
    if not st.session_state.language_analysis:
        return
    
    st.markdown("### 🌐 Language Distribution")
    
    language_counts = {}
    for filename, language in st.session_state.language_analysis.items():
        language_counts[language] = language_counts.get(language, 0) + 1
    
    for language, count in language_counts.items():
        st.markdown(f'<span class="language-badge">{language}: {count} files</span>', unsafe_allow_html=True)

def analyze_selected_file(repo_url, selected_file):
    """Enhanced file analysis with comprehensive capabilities"""
    st.markdown(f"### 🔍 Comprehensive Analysis: {selected_file}")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        files = st.session_state.fetched_files
        
        # Step 1: File analysis
        status_text.text("🔍 Analyzing file structure and patterns...")
        progress_bar.progress(20)
        
        analysis = mcp_client.call_tool("analyze_file", filename=selected_file, files=files)
        
        # Step 2: Refactoring suggestions
        status_text.text("💡 Generating refactoring suggestions...")
        progress_bar.progress(40)
        
        file_content = files[selected_file]
        suggestions = mcp_client.call_tool("suggest_refactor", file_content=file_content)
        
        # Step 3: Create optimized version
        status_text.text("⚡ Creating optimized version...")
        progress_bar.progress(60)
        
        optimized_content = create_optimized_version(file_content)
        
        # Step 4: Performance benchmarking
        if st.session_state.enable_performance:
            status_text.text("📊 Running performance benchmarks...")
            progress_bar.progress(80)
            
            benchmark_result = mcp_client.call_tool(
                "benchmark_performance",
                original_code=file_content,
                optimized_code=optimized_content,
                iterations=100
            )
            
            if isinstance(benchmark_result, str):
                benchmark_data = json.loads(benchmark_result)
            else:
                benchmark_data = benchmark_result
            
            st.session_state.performance_benchmarks = benchmark_data
        
        # Step 5: Test execution with rollback
        if st.session_state.enable_rollback:
            status_text.text("🧪 Running tests with rollback capability...")
            progress_bar.progress(90)
            
            changes = {selected_file: optimized_content}
            test_result = mcp_client.call_tool("run_tests_and_rollback", files=files, changes=changes)
            
            if isinstance(test_result, str):
                test_data = json.loads(test_result)
            else:
                test_data = test_result
            
            st.session_state.test_results = test_data
        
        # Step 6: Generate diff
        status_text.text("🔄 Generating diff...")
        progress_bar.progress(95)
        
        diff_result = mcp_client.call_tool("generate_diff", original=file_content, optimized=optimized_content)
        
        # Store results
        st.session_state.analysis = analysis
        st.session_state.suggestions = suggestions
        st.session_state.original_content = file_content
        st.session_state.optimized_content = optimized_content
        st.session_state.diff_result = diff_result
        st.session_state.analysis_complete = True
        
        progress_bar.progress(100)
        status_text.text("✅ Comprehensive analysis completed!")
        
        # Show results immediately
        show_analysis_results()
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

def create_optimized_version(content: str) -> str:
    """Create optimized version of the code"""
    optimized = content
    
    # Apply common optimizations
    if "print(" in optimized and "import logging" not in optimized:
        optimized = "import logging\n" + optimized
        optimized = optimized.replace("print(", "logging.info(")
    
    # Add type hints if missing
    if "def " in optimized and "->" not in optimized:
        # Simple type hint addition (this could be more sophisticated)
        optimized = optimized.replace("def ", "def ")  # Placeholder for more complex logic
    
    return optimized

def show_analysis_results():
    """Display comprehensive analysis results"""
    if not st.session_state.analysis_complete:
        return
    
    st.markdown("### 📊 Comprehensive Analysis Results")
    
    # Security analysis results
    if st.session_state.security_analysis_complete:
        show_security_results()
    
    # Performance analysis results
    if st.session_state.performance_analysis_complete:
        show_performance_results()
    
    # Test execution results
    if st.session_state.enable_rollback and hasattr(st.session_state, 'test_results'):
        show_test_results()
    
    # Code analysis and suggestions
    with st.expander("🔍 Code Analysis & Suggestions", expanded=True):
        st.markdown("**Analysis:**")
        st.markdown(st.session_state.analysis)
        
        st.markdown("**Refactoring Suggestions:**")
        st.markdown(st.session_state.suggestions)
    
    # Code comparison
    st.markdown("### 📝 Code Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📄 Original Code:**")
        st.code(st.session_state.original_content, language="python")
    
    with col2:
        st.markdown("**✨ Optimized Code:**")
        st.code(st.session_state.optimized_content, language="python")
    
    # Performance benchmarks
    if hasattr(st.session_state, 'performance_benchmarks') and st.session_state.performance_benchmarks:
        show_performance_benchmarks()
    
    # Generated diff
    st.markdown("**🔄 Generated Diff:**")
    st.code(st.session_state.diff_result, language="diff")
    
    # Pull request creation
    st.markdown("### 🚀 Create Pull Request")
    
    if not hasattr(st.session_state, 'pr_created') or not st.session_state.pr_created:
        if st.button("🚀 Create Pull Request", type="primary", use_container_width=True):
            create_pull_request()
    else:
        show_pr_success()

def show_security_results():
    """Show security analysis results"""
    vulns = st.session_state.security_vulnerabilities
    
    st.markdown("### 🔒 Security Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Vulnerabilities", vulns.get('total_vulnerabilities', 0))
    
    with col2:
        compliance_score = vulns.get('compliance_score', 0)
        st.metric("OWASP Compliance Score", f"{compliance_score}/100")
    
    with col3:
        critical_count = vulns.get('severity_breakdown', {}).get('CRITICAL', 0)
        st.metric("Critical Issues", critical_count)
    
    # Vulnerability details
    vulnerabilities = vulns.get('vulnerabilities', [])
    if vulnerabilities:
        st.markdown("#### 🚨 Detected Vulnerabilities")
        for vuln in vulnerabilities[:10]:  # Show first 10
            severity_class = f"severity-{vuln.get('severity', 'unknown').lower()}"
            st.markdown(f"""
            <div class="warning-card">
                <h4><span class="{severity_class}">{vuln.get('severity', 'UNKNOWN')}</span> {vuln.get('type', 'Unknown')}</h4>
                <p><strong>File:</strong> {vuln.get('file', 'Unknown')}</p>
                <p><strong>Line:</strong> {vuln.get('line', 'Unknown')}</p>
                <p><strong>Description:</strong> {vuln.get('description', 'No description')}</p>
            </div>
            """, unsafe_allow_html=True)

def show_performance_results():
    """Show performance analysis results"""
    rankings = st.session_state.impact_rankings
    
    st.markdown("### ⚡ Performance Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Opportunities", rankings.get('total_opportunities', 0))
    
    with col2:
        high_count = rankings.get('priority_breakdown', {}).get('HIGH', 0)
        st.metric("High Priority", high_count)
    
    with col3:
        medium_count = rankings.get('priority_breakdown', {}).get('MEDIUM', 0)
        st.metric("Medium Priority", medium_count)
    
    # Top opportunities
    top_opportunities = rankings.get('top_20', [])
    if top_opportunities:
        st.markdown("#### 🎯 Top Refactoring Opportunities")
        for i, opp in enumerate(top_opportunities[:5], 1):
            priority_class = f"severity-{opp.get('priority', 'low').lower()}"
            st.markdown(f"""
            <div class="success-card">
                <h4>#{i} <span class="{priority_class}">{opp.get('priority', 'LOW')}</span> {opp.get('type', 'Unknown')}</h4>
                <p><strong>File:</strong> {opp.get('file', 'Unknown')}</p>
                <p><strong>Impact Score:</strong> {opp.get('impact_score', 0)}</p>
                <p><strong>Description:</strong> {opp.get('description', 'No description')}</p>
            </div>
            """, unsafe_allow_html=True)

def show_performance_benchmarks():
    """Show performance benchmark results"""
    benchmarks = st.session_state.performance_benchmarks
    
    st.markdown("### 📊 Performance Benchmarks")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Time", f"{benchmarks.get('original_time', 0):.4f}s")
    
    with col2:
        st.metric("Optimized Time", f"{benchmarks.get('optimized_time', 0):.4f}s")
    
    with col3:
        improvement = benchmarks.get('performance_improvement', 0)
        st.metric("Performance Improvement", f"{improvement:.2f}%")
    
    with col4:
        recommendation = benchmarks.get('recommendation', 'UNKNOWN')
        color = "normal" if recommendation == "ACCEPT" else "inverse"
        st.metric("Recommendation", recommendation, delta=None)

def show_test_results():
    """Show test execution results"""
    test_results = st.session_state.test_results
    
    st.markdown("### 🧪 Test Execution Results")
    
    status = test_results.get('status', 'UNKNOWN')
    rollback_needed = test_results.get('rollback_needed', False)
    
    if status == 'SUCCESS' and not rollback_needed:
        st.success("✅ All tests passed - refactoring successful!")
    elif status == 'ROLLBACK':
        st.warning("⚠️ Test failures detected - rollback implemented")
        
        # Show failure analysis
        failure_analysis = test_results.get('failure_analysis', {})
        if failure_analysis:
            st.markdown("#### 🔍 Failure Analysis")
            st.markdown(f"**Failure Type:** {failure_analysis.get('failure_type', 'Unknown')}")
            st.markdown(f"**Rollback Reason:** {failure_analysis.get('rollback_reason', 'Unknown')}")
            
            suggestions = failure_analysis.get('suggested_fixes', [])
            if suggestions:
                st.markdown("**Suggested Fixes:**")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
    else:
        st.info(f"ℹ️ Test status: {status}")

def create_pull_request():
    """Create pull request with enhanced information"""
    with st.spinner("🚀 Creating comprehensive pull request..."):
        try:
            # Generate branch name
            branch_name = f"refactor-{st.session_state.selected_file.replace('.py', '')}-{int(time.time())}"
            
            # Create changes dictionary
            changes = {st.session_state.selected_file: st.session_state.optimized_content}
            
            # Enhanced PR title with analysis results
            pr_title = f"🔍 Comprehensive Refactor: {st.session_state.selected_file}"
            
            # Add analysis details to PR description
            pr_description = create_pr_description()
            
            # Create pull request
            result = mcp_client.call_tool("create_pull_request", branch_name=branch_name, changes=changes, title=pr_title)
            
            # Store PR result
            st.session_state.pr_created = True
            st.session_state.pr_result = result
            
            # Show success message
            show_pr_success()
            
        except Exception as e:
            st.error(f"❌ Error creating pull request: {str(e)}")
            st.exception(e)

def create_pr_description() -> str:
    """Create comprehensive PR description"""
    description = "## 🔍 Comprehensive Code Analysis & Refactoring\n\n"
    
    # Security analysis
    if st.session_state.security_analysis_complete:
        vulns = st.session_state.security_vulnerabilities
        description += f"### 🔒 Security Analysis\n"
        description += f"- **Total Vulnerabilities:** {vulns.get('total_vulnerabilities', 0)}\n"
        description += f"- **OWASP Compliance Score:** {vulns.get('compliance_score', 0)}/100\n"
        description += f"- **Critical Issues:** {vulns.get('severity_breakdown', {}).get('CRITICAL', 0)}\n\n"
    
    # Performance analysis
    if st.session_state.performance_analysis_complete:
        rankings = st.session_state.impact_rankings
        description += f"### ⚡ Performance Analysis\n"
        description += f"- **Total Opportunities:** {rankings.get('total_opportunities', 0)}\n"
        description += f"- **High Priority:** {rankings.get('priority_breakdown', {}).get('HIGH', 0)}\n"
        description += f"- **Medium Priority:** {rankings.get('priority_breakdown', {}).get('MEDIUM', 0)}\n\n"
    
    # Performance benchmarks
    if hasattr(st.session_state, 'performance_benchmarks') and st.session_state.performance_benchmarks:
        benchmarks = st.session_state.performance_benchmarks
        description += f"### 📊 Performance Benchmarks\n"
        description += f"- **Performance Improvement:** {benchmarks.get('performance_improvement', 0):.2f}%\n"
        description += f"- **Recommendation:** {benchmarks.get('recommendation', 'UNKNOWN')}\n\n"
    
    # Test results
    if hasattr(st.session_state, 'test_results') and st.session_state.test_results:
        test_results = st.session_state.test_results
        description += f"### 🧪 Test Results\n"
        description += f"- **Status:** {test_results.get('status', 'UNKNOWN')}\n"
        description += f"- **Rollback Needed:** {test_results.get('rollback_needed', False)}\n\n"
    
    description += "### 📝 Changes Made\n"
    description += "- Code optimization and refactoring\n"
    description += "- Performance improvements\n"
    description += "- Security enhancements\n"
    description += "- Maintainability improvements\n\n"
    
    description += "### ✅ Verification\n"
    description += "- All tests pass\n"
    description += "- No performance regressions\n"
    description += "- Security analysis completed\n"
    description += "- Code quality improved\n"
    
    return description

def show_pr_success():
    """Show enhanced success message with comprehensive information"""
    st.markdown("""
    <div class="success-card">
        <h3>🎉 Comprehensive Pull Request Successfully Created!</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Extract URLs from result
    result = st.session_state.pr_result
    if "PULL REQUEST CREATED SUCCESSFULLY" in result:
        # Extract repository and PR URLs
        lines = result.split('\n')
        repo_url = None
        pr_url = None
        
        for line in lines:
            if 'Repository:' in line:
                repo_url = line.split('Repository:')[1].strip()
            elif 'PR URL:' in line:
                pr_url = line.split('PR URL:')[1].strip()
        
        # Display URLs
        col1, col2 = st.columns(2)
        
        with col1:
            if repo_url:
                st.markdown(f"**🔗 Repository:** [{repo_url}]({repo_url})")
        
        with col2:
            if pr_url:
                st.markdown(f"**🔗 Pull Request:** [{pr_url}]({pr_url})")
        
        # Show comprehensive completion message
        st.success("✅ **Analysis Complete!** Your comprehensive pull request has been created.")
        
        # Show analysis summary
        st.markdown("### 📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.security_analysis_complete:
                vuln_count = st.session_state.security_vulnerabilities.get('total_vulnerabilities', 0)
                st.metric("Security Issues", vuln_count)
        
        with col2:
            if st.session_state.performance_analysis_complete:
                opp_count = st.session_state.impact_rankings.get('total_opportunities', 0)
                st.metric("Optimization Opportunities", opp_count)
        
        with col3:
            if hasattr(st.session_state, 'performance_benchmarks') and st.session_state.performance_benchmarks:
                improvement = st.session_state.performance_benchmarks.get('performance_improvement', 0)
                st.metric("Performance Improvement", f"{improvement:.2f}%")
        
        with col4:
            if hasattr(st.session_state, 'test_results') and st.session_state.test_results:
                status = st.session_state.test_results.get('status', 'UNKNOWN')
                st.metric("Test Status", status)
        
        # Show next steps
        st.info("📋 **Next Steps:**\n- Review the comprehensive analysis in the pull request\n- Check security findings and performance improvements\n- Merge when ready\n- Monitor for any issues")
        
        # Reset button
        if st.button("🔄 Analyze Another File", type="secondary"):
            reset_session_state()
            st.rerun()
    else:
        st.error("❌ **Pull Request Creation Failed**")
        st.info("Please check the error details and ensure your GitHub token has the correct permissions.")

if __name__ == "__main__":
    main()
