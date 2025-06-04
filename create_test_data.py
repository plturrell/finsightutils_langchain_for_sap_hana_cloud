#!/usr/bin/env python3
"""
This script helps create test data for the SAP HANA Cloud LangChain integration.
It can be used to generate sample text documents, embeddings, and test queries
that can be used to validate the functionality of the system once access is available.
"""

import json
import numpy as np
import os
from typing import List, Dict, Any, Optional

class TestDataGenerator:
    def __init__(self, output_dir: str = "./test_data"):
        """Initialize the test data generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_documents(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate sample documents with metadata"""
        documents = []
        
        # Document templates for SAP HANA Cloud and LangChain integration
        templates = [
            "SAP HANA Cloud offers {feature} capabilities that help businesses {benefit}.",
            "Using LangChain with SAP HANA Cloud enables developers to {action} which {outcome}.",
            "The integration between SAP HANA Cloud and LangChain provides {advantage} for {use_case} scenarios.",
            "SAP HANA Cloud's {component} works with LangChain's {lc_component} to enable {capability}.",
            "When implementing {solution} with SAP HANA Cloud and LangChain, it's important to consider {consideration}."
        ]
        
        # Fill-in values for templates
        features = ["vector search", "graph database", "in-memory", "columnar storage", "multi-model", "spatial", "predictive analytics"]
        benefits = ["gain insights faster", "reduce operational costs", "improve decision making", "enhance customer experiences", "streamline operations"]
        actions = ["build RAG applications", "create knowledge graph-based agents", "implement semantic search", "develop conversational AI", "build multimodal applications"]
        outcomes = ["improves accuracy of responses", "reduces development time", "enhances data security", "provides better business insights", "scales efficiently"]
        advantages = ["seamless data access", "high-performance vector operations", "enterprise-grade security", "simplified deployment", "GPU acceleration"]
        use_cases = ["customer support", "document analysis", "knowledge management", "product recommendation", "fraud detection"]
        components = ["vector engine", "graph engine", "spatial engine", "text analysis", "machine learning integration"]
        lc_components = ["vectorstores", "retrievers", "agents", "memory", "chains"]
        capabilities = ["real-time insights", "context-aware responses", "semantic understanding", "data-driven decision making", "personalized experiences"]
        solutions = ["RAG architectures", "agent workflows", "question answering systems", "knowledge bases", "chatbots"]
        considerations = ["data privacy", "performance optimization", "data quality", "scalability", "user experience"]
        
        # Generate documents
        for i in range(count):
            template_idx = i % len(templates)
            template = templates[template_idx]
            
            # Create document based on template
            if template_idx == 0:
                text = template.format(
                    feature=np.random.choice(features),
                    benefit=np.random.choice(benefits)
                )
            elif template_idx == 1:
                text = template.format(
                    action=np.random.choice(actions),
                    outcome=np.random.choice(outcomes)
                )
            elif template_idx == 2:
                text = template.format(
                    advantage=np.random.choice(advantages),
                    use_case=np.random.choice(use_cases)
                )
            elif template_idx == 3:
                text = template.format(
                    component=np.random.choice(components),
                    lc_component=np.random.choice(lc_components),
                    capability=np.random.choice(capabilities)
                )
            else:
                text = template.format(
                    solution=np.random.choice(solutions),
                    consideration=np.random.choice(considerations)
                )
            
            # Add metadata
            metadata = {
                "id": f"doc_{i}",
                "category": np.random.choice(["technical", "business", "tutorial", "case_study", "reference"]),
                "tags": np.random.choice(["vector", "graph", "rag", "langchain", "embedding"], 
                                        size=np.random.randint(1, 4), 
                                        replace=False).tolist(),
                "source": np.random.choice(["documentation", "blog", "whitepaper", "code_example", "user_guide"]),
                "relevance": np.random.randint(1, 11)
            }
            
            documents.append({
                "page_content": text,
                "metadata": metadata
            })
        
        # Save documents to file
        with open(os.path.join(self.output_dir, "sample_documents.json"), "w") as f:
            json.dump(documents, f, indent=2)
            
        print(f"Generated {len(documents)} sample documents")
        return documents
    
    def generate_test_queries(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate test queries for the vector store"""
        queries = [
            "How does SAP HANA Cloud vector search work with LangChain?",
            "What are the benefits of using SAP HANA Cloud with LangChain?",
            "Can you explain the graph capabilities in SAP HANA Cloud?",
            "How do I implement RAG with SAP HANA Cloud and LangChain?",
            "What performance optimizations are available for vector search?",
            "How does the HNSW index work in SAP HANA Cloud?",
            "What is the difference between cosine similarity and euclidean distance?",
            "How can I filter metadata in SAP HANA vector searches?",
            "What is maximal marginal relevance and how does it work?",
            "How do I use internal embeddings in SAP HANA Cloud?",
            "What GPU acceleration features are available?",
            "How can I deploy the SAP HANA Cloud LangChain integration?",
            "What are the best practices for optimizing vector search performance?",
            "How does context-aware error handling work in the integration?",
            "What is the difference between internal and external embeddings?",
        ]
        
        # Select random subset if count < len(queries)
        if count < len(queries):
            selected_queries = np.random.choice(queries, size=count, replace=False).tolist()
        else:
            selected_queries = queries
        
        # Create test query objects
        test_queries = []
        for i, query in enumerate(selected_queries):
            test_query = {
                "id": f"query_{i}",
                "text": query,
                "options": {
                    "k": np.random.choice([3, 5, 10]),
                    "filter": self._generate_random_filter() if np.random.random() > 0.5 else None,
                    "use_mmr": np.random.random() > 0.7,
                }
            }
            
            # Add lambda parameter if using MMR
            if test_query["options"]["use_mmr"]:
                test_query["options"]["lambda_mult"] = round(np.random.uniform(0.3, 0.9), 1)
                
            test_queries.append(test_query)
        
        # Save queries to file
        with open(os.path.join(self.output_dir, "test_queries.json"), "w") as f:
            json.dump(test_queries, f, indent=2)
            
        print(f"Generated {len(test_queries)} test queries")
        return test_queries
    
    def _generate_random_filter(self) -> Dict[str, Any]:
        """Generate a random metadata filter"""
        filter_types = ["simple", "complex", "nested"]
        filter_type = np.random.choice(filter_types)
        
        if filter_type == "simple":
            categories = ["technical", "business", "tutorial", "case_study", "reference"]
            return {"category": np.random.choice(categories)}
        
        elif filter_type == "complex":
            tags = ["vector", "graph", "rag", "langchain", "embedding"]
            return {"tags": np.random.choice(tags)}
        
        else:  # nested
            sources = ["documentation", "blog", "whitepaper", "code_example", "user_guide"]
            relevance_op = np.random.choice(["$gt", "$gte", "$lt", "$lte"])
            relevance_value = np.random.randint(1, 10)
            
            return {
                "$and": [
                    {"source": np.random.choice(sources)},
                    {"relevance": {relevance_op: relevance_value}}
                ]
            }
    
    def generate_mock_performance_data(self) -> Dict[str, Any]:
        """Generate mock performance data for NVIDIA T4 GPU tests"""
        # Create baseline CPU performance metrics
        cpu_metrics = {
            "embedding_generation": {
                "batch_size_1": {"latency_ms": round(np.random.uniform(40, 60), 2)},
                "batch_size_8": {"latency_ms": round(np.random.uniform(280, 320), 2)},
                "batch_size_32": {"latency_ms": round(np.random.uniform(1100, 1300), 2)},
                "batch_size_64": {"latency_ms": round(np.random.uniform(2200, 2600), 2)},
                "throughput_docs_per_sec": round(np.random.uniform(18, 25), 2)
            },
            "similarity_search": {
                "simple_query": {"latency_ms": round(np.random.uniform(30, 50), 2)},
                "filtered_query": {"latency_ms": round(np.random.uniform(50, 80), 2)},
                "mmr_search": {"latency_ms": round(np.random.uniform(100, 150), 2)}
            }
        }
        
        # Create T4 GPU performance metrics (faster than CPU)
        gpu_speedup = np.random.uniform(3.5, 5.5)
        t4_metrics = {
            "embedding_generation": {
                "batch_size_1": {"latency_ms": round(cpu_metrics["embedding_generation"]["batch_size_1"]["latency_ms"] / gpu_speedup, 2)},
                "batch_size_8": {"latency_ms": round(cpu_metrics["embedding_generation"]["batch_size_8"]["latency_ms"] / gpu_speedup, 2)},
                "batch_size_32": {"latency_ms": round(cpu_metrics["embedding_generation"]["batch_size_32"]["latency_ms"] / gpu_speedup, 2)},
                "batch_size_64": {"latency_ms": round(cpu_metrics["embedding_generation"]["batch_size_64"]["latency_ms"] / gpu_speedup, 2)},
                "batch_size_128": {"latency_ms": round(cpu_metrics["embedding_generation"]["batch_size_64"]["latency_ms"] * 1.5 / gpu_speedup, 2)},
                "throughput_docs_per_sec": round(cpu_metrics["embedding_generation"]["throughput_docs_per_sec"] * gpu_speedup, 2)
            },
            "similarity_search": {
                "simple_query": {"latency_ms": round(cpu_metrics["similarity_search"]["simple_query"]["latency_ms"] / 1.2, 2)},
                "filtered_query": {"latency_ms": round(cpu_metrics["similarity_search"]["filtered_query"]["latency_ms"] / 1.2, 2)},
                "mmr_search": {"latency_ms": round(cpu_metrics["similarity_search"]["mmr_search"]["latency_ms"] / 3.0, 2)}
            },
            "gpu_utilization": {
                "memory_usage_mb": round(np.random.uniform(2000, 4000), 0),
                "compute_utilization_pct": round(np.random.uniform(40, 85), 1),
                "max_batch_size": 128
            }
        }
        
        performance_data = {
            "cpu_metrics": cpu_metrics,
            "t4_gpu_metrics": t4_metrics,
            "speedup_factor": {
                "embedding_generation": round(gpu_speedup, 2),
                "mmr_search": 3.0,
                "overall": round((gpu_speedup + 3.0) / 2, 2)
            }
        }
        
        # Save performance data to file
        with open(os.path.join(self.output_dir, "performance_data.json"), "w") as f:
            json.dump(performance_data, f, indent=2)
            
        print(f"Generated mock performance data")
        return performance_data
        
    def generate_all(self):
        """Generate all test data"""
        self.generate_documents()
        self.generate_test_queries()
        self.generate_mock_performance_data()
        print(f"All test data generated in {self.output_dir}")


if __name__ == "__main__":
    generator = TestDataGenerator()
    generator.generate_all()