#!/usr/bin/env python3
"""
CLI script to analyze agent code in langchain-integration-for-sap-hana-cloud.
This script provides a command-line interface to the code analyzer.
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add source directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Try to import from tools directory
try:
    from tools.code_analyzer import analyze_codebase
except ImportError:
    # Add shared directory to Python path
    shared_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared')
    if shared_dir not in sys.path:
        sys.path.append(shared_dir)
    
    try:
        from finsight_code_analyzer import analyze_project_code
        
        async def analyze_codebase(project_path=None, use_nvidia=True, max_workers=8, output_file=None):
            config = {
                "use_nvidia": use_nvidia,
                "max_workers": max_workers
            }
            if project_path is None:
                project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return await analyze_project_code(config, project_path)
    except ImportError:
        print("Error: Could not import code analyzer. Make sure it's available in the shared directory.")
        sys.exit(1)

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze code to map files to agent functionality")
    parser.add_argument("--output", "-o", help="Output file path for analysis results (JSON)")
    parser.add_argument("--no-nvidia", action="store_true", help="Don't use NVIDIA LLM for code analysis")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of workers for parallel processing")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of results")
    parser.add_argument("--project-path", help="Path to project to analyze (defaults to current project)")
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"agent_analysis_{timestamp}.json"
    
    print(f"Analyzing langchain-integration-for-sap-hana-cloud code...")
    print(f"This may take a few minutes...")
    
    # Run analysis
    start_time = datetime.now()
    results = await analyze_codebase(
        project_path=args.project_path,
        use_nvidia=not args.no_nvidia,
        max_workers=args.max_workers,
        output_file=args.output
    )
    end_time = datetime.now()
    
    # Print summary
    print(f"\nAnalysis completed in {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"Found {len(results['analysis_results'])} agent-related files")
    
    # Print detected patterns
    if results.get('patterns'):
        print("\nDetected patterns:")
        for agent_type, pattern in results['patterns'].items():
            print(f"  - {agent_type} ({pattern['file_count']} files)")
            if pattern.get('common_capabilities'):
                print(f"    Common capabilities: {', '.join(pattern['common_capabilities'])}")
    
    # Print improvement statistics
    suggestion_count = sum(len(result.get('improvement_suggestions', [])) for result in results['analysis_results'])
    if suggestion_count > 0:
        print(f"\nFound {suggestion_count} improvement suggestions")
        
        # Group by severity
        severity_counts = {}
        for result in results['analysis_results']:
            for suggestion in result.get('improvement_suggestions', []):
                severity = suggestion.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            print(f"  - {severity}: {count}")
    
    print(f"\nResults saved to {args.output}")
    
    # Generate visualization if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create dependency graph visualization
            print("Generating dependency graph visualization...")
            
            # Extract nodes and edges from results
            nodes = results['dependency_graph']['central_nodes']
            
            # Create a simplified graph for visualization
            if nodes:
                plt.figure(figsize=(12, 8))
                G = nx.DiGraph()
                
                # Add nodes
                for node, centrality in nodes:
                    G.add_node(node, centrality=centrality)
                
                # Try to add some edges if available
                for result in results['analysis_results']:
                    file_path = result['file_path']
                    dependencies = result.get('dependencies', [])
                    for dep in dependencies:
                        if dep in G.nodes:
                            G.add_edge(file_path, dep)
                
                # Draw the graph
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                        node_size=[v * 5000 + 100 for v in nx.get_node_attributes(G, 'centrality').values()], 
                        edge_color='gray', linewidths=1, font_size=10)
                
                # Save the visualization
                viz_file = args.output.replace('.json', '_dependency_graph.png')
                plt.savefig(viz_file)
                plt.close()
                
                print(f"Dependency graph saved to {viz_file}")
            else:
                print("No central nodes found for visualization")
        except ImportError:
            print("Visualization requires matplotlib and networkx. Install with: pip install matplotlib networkx")

if __name__ == "__main__":
    asyncio.run(main())
