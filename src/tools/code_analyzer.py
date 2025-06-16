"""
Code analyzer integration for langchain-integration-for-sap-hana-cloud.
This module provides an interface to the shared code analyzer.
"""

import os
import sys
import asyncio
from typing import Dict, Any, List, Optional

# Add shared directory to Python path
shared_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'shared')
if shared_dir not in sys.path:
    sys.path.append(shared_dir)

try:
    from finsight_code_analyzer import analyze_project_code, CodeAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("Warning: Code analyzer not found in shared directory")

async def analyze_codebase(
    project_path: Optional[str] = None,
    use_nvidia: bool = True,
    max_workers: int = 8,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the project codebase to map code to agent functionality.
    
    Args:
        project_path: Path to the project root directory (defaults to current project)
        use_nvidia: Whether to use NVIDIA LLM for analysis
        max_workers: Maximum number of workers for parallel processing
        output_file: Optional path to save results to JSON file
        
    Returns:
        Dictionary with analysis results
    """
    if not ANALYZER_AVAILABLE:
        return {"error": "Code analyzer not available"}
    
    # Use current project path if not specified
    if project_path is None:
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Configure analyzer
    config = {
        "use_nvidia": use_nvidia,
        "max_workers": max_workers,
        "excluded_dirs": ["venv", "node_modules", "__pycache__", ".git", "dist", "build"],
        "file_patterns": ["**/*.py", "**/*.tsx", "**/*.jsx", "**/*.ts", "**/*.js"]
    }
    
    # Run analysis
    results = await analyze_project_code(config, project_path)
    
    # Save results if output file specified
    if output_file:
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save results to file
        with open(output_file, "w") as f:
            # Handle CodeDefinition objects
            def serialize(obj):
                if hasattr(obj, "to_dict"):
                    return obj.to_dict()
                return str(obj)
            
            json.dump(results, f, indent=2, default=serialize)
    
    return results

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze code to map files to agent functionality")
    parser.add_argument("--output", "-o", help="Output file path for analysis results (JSON)")
    parser.add_argument("--no-nvidia", action="store_true", help="Don't use NVIDIA LLM for code analysis")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of workers for parallel processing")
    
    args = parser.parse_args()
    
    # Run analysis
    asyncio.run(analyze_codebase(
        use_nvidia=not args.no_nvidia,
        max_workers=args.max_workers,
        output_file=args.output
    ))
