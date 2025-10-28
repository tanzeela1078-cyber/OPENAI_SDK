#!/usr/bin/env python3


import asyncio
import os
import sys
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.mcp.server import MCPServerStdio

# ──────────────────────────────────────────────────────────────
# 🔧 Configuration
# ──────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  Warning: OPENAI_API_KEY not found in .env — server may fail if it requires it.")

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT = os.path.join(HERE, "mcp_server_stdio.py")

if not os.path.exists(SERVER_SCRIPT):
    raise FileNotFoundError(f"❌ MCP server not found at: {SERVER_SCRIPT}")

PYTHON_EXEC = sys.executable
CLIENT_SESSION_TIMEOUT_SECONDS = 30.0

# ──────────────────────────────────────────────────────────────
# 🧠 OpenAI model configuration
# ──────────────────────────────────────────────────────────────
external_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
)
model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=external_client,
)

# ──────────────────────────────────────────────────────────────
# 🚀 Agent launcher
# ──────────────────────────────────────────────────────────────
async def run_agent_interaction():
    params = {"command": PYTHON_EXEC, "args": [SERVER_SCRIPT]}

    print("⏳ Launching MCP Server via:", PYTHON_EXEC, SERVER_SCRIPT)

    async with MCPServerStdio(
        params=params,
        client_session_timeout_seconds=CLIENT_SESSION_TIMEOUT_SECONDS,
        name="StaticAnalysisServer",
    ) as mcp_server:
        print("✅ MCP server connected and ready.\n")

        # Agent with integrated MCP tools
        agent = Agent(
            name="AuditorAgent",
            instructions=(
                "You are an advanced code analysis and refactoring agent with comprehensive capabilities. "
                "You can analyze multi-language codebases, detect security vulnerabilities, benchmark performance, "
                "and implement rollback mechanisms. You support Python, JavaScript, TypeScript, Go, Rust, Java, "
                "C++, C#, PHP, Ruby, Swift, and Kotlin. "
                "IMPORTANT: When using analyze_file or suggest_refactor tools, you MUST pass both 'filename' and 'files' parameters. "
                "The 'files' parameter should contain the dictionary returned from fetch_files. "
                "Always store the result from fetch_files and pass it to subsequent analysis tools. "
                "For large codebases, use chunking support. Always run security analysis and performance benchmarks. "
                "Implement automatic rollback if tests fail. Rank opportunities by impact score."
            ),
            mcp_servers=[mcp_server],
        )

        
        # Get repository URL from environment
        repo_url = os.getenv("REPO_URL")
        
        initial_input = (
            f"STEP 1: Fetch the codebase from {repo_url} using fetch_files tool with chunking support. "
            f"STEP 2: Use display_fetched_files tool to show all fetched files with their content. "
            f"STEP 3: Run comprehensive security analysis using analyze_security_vulnerabilities tool. "
            f"STEP 4: Rank refactoring opportunities using rank_refactoring_opportunities tool to identify top 20 opportunities. "
            f"STEP 5: Analyze specific files using analyze_file tool. IMPORTANT: You must pass the 'files' parameter from the fetch_files result to analyze_file. "
            f"STEP 6: Generate refactoring suggestions using suggest_refactor tool. IMPORTANT: You must pass the 'files' parameter from the fetch_files result to suggest_refactor. "
            f"STEP 7: Create optimized versions and use generate_diff tool to show the differences. "
            f"STEP 8: Benchmark performance using benchmark_performance tool to ensure no regressions. "
            f"STEP 9: Run tests with rollback capability using run_tests_and_rollback tool. "
            f"STEP 10: Use display_diffs tool to show all generated diffs. "
            f"STEP 11: Use get_session_data tool to show complete session information. "
            f"STEP 12: Create a pull request with the suggested changes using create_pull_request tool. "
            f"Make sure to display all fetched files, analysis results, security findings, performance benchmarks, and diffs visibly in your response. "
            f"CRITICAL: When calling analyze_file or suggest_refactor, you MUST include both 'filename' and 'files' parameters from the previous fetch_files result. "
            f"Ensure all security vulnerabilities are detected and classified by severity. "
            f"Verify performance benchmarks show no regressions before proceeding. "
            f"Implement automatic rollback if tests fail."
        )

        print("🚀 Running agent...")
        print("📋 Task: Fetching and analyzing GitHub repository...")
        
        result = await Runner.run(agent, initial_input)

        print("\n" + "="*60)
        print("🎯 AGENT EXECUTION COMPLETED")
        print("="*60)
        print("\n📊 **Final Output:**")
        print(result.final_output if hasattr(result, "final_output") else result)
        print("\n" + "="*60)
        print("✅ Task completed successfully!")
        print("="*60 + "\n")

    print("🔒 MCP server stopped and cleaned up.")


def main():
    try:
        asyncio.run(run_agent_interaction())
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
    except Exception as e:
        print(f"❌ Unexpected error in agent: {e}")
        raise


if __name__ == "__main__":
    main()
