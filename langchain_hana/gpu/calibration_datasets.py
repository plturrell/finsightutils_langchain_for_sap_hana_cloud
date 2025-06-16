"""
Enhanced calibration datasets for INT8 quantization.

This module provides domain-specific calibration datasets for INT8 quantization
to improve accuracy across different types of text content.
"""

from typing import List, Dict, Any, Optional, Union
import logging
import os
import json
import random

# Import specialized financial calibration if available
try:
    from langchain_hana.gpu.financial_calibration import get_financial_calibration_dataset
    HAS_FINANCIAL_CALIBRATION = True
except ImportError:
    HAS_FINANCIAL_CALIBRATION = False

logger = logging.getLogger(__name__)

# Common calibration texts covering different domains
GENERAL_CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "TensorRT provides high-performance inference for deep learning models.",
    "The Eiffel Tower is located in Paris, France.",
    "Quantum computing leverages quantum mechanics to process information.",
    "Natural language processing enables computers to understand human language.",
    "The human genome contains approximately 3 billion base pairs.",
    "Climate change is affecting ecosystems worldwide.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "Neural networks consist of layers of interconnected nodes.",
    "The Internet of Things connects everyday devices to the internet.",
    "Cloud computing provides on-demand access to computing resources.",
    "Cybersecurity is essential for protecting digital systems from attacks.",
    "Blockchain technology enables secure and transparent transactions.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
    "Virtual reality creates immersive digital environments.",
    "The theory of relativity was developed by Albert Einstein.",
    "Data science combines statistics, mathematics, and programming.",
    "Embedded systems are specialized computing systems with dedicated functions.",
    "Autonomous vehicles use sensors and AI to navigate without human input.",
]

# Financial domain calibration texts
FINANCIAL_CALIBRATION_TEXTS = [
    "The quarterly earnings report showed a 15% increase in revenue.",
    "Market volatility increased due to geopolitical tensions.",
    "The central bank raised interest rates by 25 basis points.",
    "Investors are concerned about inflation affecting asset values.",
    "The company's stock price fell after the disappointing earnings call.",
    "Cryptocurrency markets experienced significant fluctuations this month.",
    "Risk management strategies help mitigate potential financial losses.",
    "The merger resulted in a combined market capitalization of $50 billion.",
    "ESG criteria are increasingly important for investment decisions.",
    "The balance sheet shows strong liquidity and manageable debt levels.",
    "Capital expenditures increased to support the company's growth strategy.",
    "Dividend yields vary across different industry sectors.",
    "The price-to-earnings ratio is a common valuation metric for stocks.",
    "Asset allocation is crucial for managing investment portfolio risk.",
    "The quarterly financial statement complies with GAAP standards.",
    "Treasury bonds are considered low-risk fixed-income investments.",
    "Foreign exchange markets operate 24 hours a day, five days a week.",
    "Algorithmic trading uses computer programs to execute trades automatically.",
    "The yield curve inverted, potentially signaling a future recession.",
    "Private equity funds typically have a long-term investment horizon.",
]

# Enterprise SAP domain calibration texts
SAP_CALIBRATION_TEXTS = [
    "SAP HANA Cloud provides in-memory database capabilities for enterprise applications.",
    "The S/4HANA system centralizes ERP functionality in a single platform.",
    "SAP Business Technology Platform enables custom application development.",
    "SAP Fiori provides a modern user experience for SAP applications.",
    "Integration between SAP systems uses standard APIs and protocols.",
    "SAP BW/4HANA is the next generation of SAP's data warehousing solution.",
    "SAP Analytics Cloud combines BI, planning, and predictive analytics.",
    "SAP SuccessFactors offers comprehensive HR management capabilities.",
    "SAP Cloud ALM provides application lifecycle management for cloud solutions.",
    "The SAP NetWeaver technology stack supports enterprise applications.",
    "SAP ABAP is a high-level programming language created by SAP.",
    "SAP CPI facilitates integration between cloud and on-premises systems.",
    "SAP Ariba provides procurement and supply chain collaboration solutions.",
    "SAP Concur streamlines travel and expense management processes.",
    "SAP IBP enables integrated business planning across the enterprise.",
    "SAP HANA uses columnar storage for efficient data processing.",
    "SAP Leonardo incorporates AI and machine learning capabilities.",
    "SAP BTP Kyma extends SAP's cloud-native application capabilities.",
    "SAP BTP ABAP Environment enables ABAP development in the cloud.",
    "SAP Data Intelligence orchestrates data processing and analytics workflows.",
]

# Technical domain calibration texts
TECHNICAL_CALIBRATION_TEXTS = [
    "The REST API uses JSON for data interchange between client and server.",
    "Kubernetes orchestrates containerized applications across multiple hosts.",
    "Docker containers package applications with their dependencies.",
    "TensorFlow and PyTorch are popular deep learning frameworks.",
    "GraphQL provides a query language for API data retrieval.",
    "Microservices architecture breaks applications into loosely coupled services.",
    "CI/CD pipelines automate software testing and deployment.",
    "Object-oriented programming uses classes and objects to structure code.",
    "Functional programming emphasizes immutable data and pure functions.",
    "Serverless computing abstracts infrastructure management from developers.",
    "Big data technologies process and analyze large volumes of data.",
    "Reactive programming focuses on asynchronous data streams.",
    "NoSQL databases provide flexible schema options for different data models.",
    "HTTPS encrypts data transmitted between web browsers and servers.",
    "OAuth 2.0 is a standard protocol for authorization.",
    "Git enables distributed version control for source code.",
    "WebAssembly allows high-performance code execution in web browsers.",
    "Edge computing processes data near the source rather than in a centralized cloud.",
    "Event-driven architecture responds to events as they occur.",
    "GraphQL APIs allow clients to request exactly the data they need.",
]

def get_domain_calibration_texts(domain: str = "general", count: int = 50) -> List[str]:
    """
    Get domain-specific calibration texts for INT8 quantization.
    
    Parameters
    ----------
    domain : str, default="general"
        Domain to get calibration texts for ("general", "financial", "sap", "technical", "all")
    count : int, default=50
        Number of texts to return for financial domain when using enhanced dataset
        
    Returns
    -------
    List[str]
        List of calibration texts
    """
    domain = domain.lower()
    if domain == "financial":
        # Use enhanced financial calibration if available
        if HAS_FINANCIAL_CALIBRATION:
            try:
                logger.info("Using enhanced financial calibration dataset")
                return get_financial_calibration_dataset(count=count)
            except Exception as e:
                logger.warning(f"Error loading enhanced financial calibration: {e}")
                logger.info("Falling back to standard financial calibration texts")
                return FINANCIAL_CALIBRATION_TEXTS
        else:
            return FINANCIAL_CALIBRATION_TEXTS
    elif domain == "sap":
        return SAP_CALIBRATION_TEXTS
    elif domain == "technical":
        return TECHNICAL_CALIBRATION_TEXTS
    elif domain == "all":
        # Combine all domains with equal representation
        all_texts = []
        all_texts.extend(GENERAL_CALIBRATION_TEXTS)
        
        # For financial, use enhanced dataset if available
        if HAS_FINANCIAL_CALIBRATION:
            try:
                financial_texts = get_financial_calibration_dataset(count=count)
                all_texts.extend(financial_texts)
            except Exception:
                all_texts.extend(FINANCIAL_CALIBRATION_TEXTS)
        else:
            all_texts.extend(FINANCIAL_CALIBRATION_TEXTS)
            
        all_texts.extend(SAP_CALIBRATION_TEXTS)
        all_texts.extend(TECHNICAL_CALIBRATION_TEXTS)
        return all_texts
    else:
        # Default to general calibration texts
        return GENERAL_CALIBRATION_TEXTS

def get_mixed_calibration_texts(domains: List[str] = None, count: int = 50) -> List[str]:
    """
    Get a mixed set of calibration texts from multiple domains.
    
    Args:
        domains: Domains to include ("general", "financial", "sap", "technical")
        count: Number of calibration texts to return
        
    Returns:
        List of mixed calibration texts
    """
    if domains is None:
        domains = ["general", "financial", "sap", "technical"]
    
    # Collect texts from all specified domains
    all_texts = []
    for domain in domains:
        all_texts.extend(get_domain_calibration_texts(domain))
    
    # Deduplicate and shuffle
    unique_texts = list(set(all_texts))
    random.shuffle(unique_texts)
    
    # Return the requested number of texts
    return unique_texts[:count]

def load_custom_calibration_texts(file_path: str) -> List[str]:
    """
    Load custom calibration texts from a file.
    
    Args:
        file_path: Path to the file containing calibration texts (one per line or JSON)
        
    Returns:
        List of calibration texts
    """
    if not os.path.exists(file_path):
        logger.warning(f"Custom calibration file {file_path} not found")
        return []
    
    try:
        # Determine file type by extension
        if file_path.endswith(".json"):
            # Load JSON file
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # List of strings
                return [str(item) for item in data if isinstance(item, (str, int, float))]
            elif isinstance(data, dict) and "texts" in data:
                # Dictionary with "texts" key
                texts = data.get("texts", [])
                return [str(item) for item in texts if isinstance(item, (str, int, float))]
            else:
                # Try to extract values from dictionary
                return [str(value) for value in data.values() if isinstance(value, (str, int, float))]
        else:
            # Load text file (one text per line)
            with open(file_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading custom calibration texts: {e}")
        return []

def create_enhanced_calibration_dataset(
    domains: List[str] = None,
    count: int = 100,
    custom_file_path: Optional[str] = None,
) -> List[str]:
    """
    Create an enhanced calibration dataset combining domain-specific and custom texts.
    
    Parameters
    ----------
    domains : List[str], optional
        Domains to include ("general", "financial", "sap", "technical")
    count : int, default=100
        Target number of calibration texts
    custom_file_path : str, optional
        Path to custom calibration text file
        
    Returns
    -------
    List[str]
        Enhanced calibration dataset
    """
    # Check if financial domain is specifically requested
    if domains is not None and "financial" in domains and len(domains) == 1:
        # Only financial domain requested - use specialized dataset
        if HAS_FINANCIAL_CALIBRATION:
            try:
                logger.info("Using specialized financial calibration dataset")
                financial_texts = get_financial_calibration_dataset(count=count)
                
                # Add custom texts if available
                custom_texts = []
                if custom_file_path:
                    custom_texts = load_custom_calibration_texts(custom_file_path)
                
                # Combine, deduplicate, and limit
                all_texts = financial_texts + custom_texts
                unique_texts = list(set(all_texts))
                random.shuffle(unique_texts)
                
                if len(unique_texts) > count:
                    unique_texts = unique_texts[:count]
                
                logger.info(f"Created enhanced financial calibration dataset with {len(unique_texts)} texts")
                return unique_texts
            except Exception as e:
                logger.warning(f"Error creating financial calibration dataset: {e}")
                logger.info("Falling back to standard mixed calibration approach")
    
    # Standard approach for mixed domains or if financial specialization failed
    domain_texts = get_mixed_calibration_texts(domains, count=count//2)
    
    # Add custom texts if available
    custom_texts = []
    if custom_file_path:
        custom_texts = load_custom_calibration_texts(custom_file_path)
    
    # Combine and deduplicate
    all_texts = domain_texts + custom_texts
    unique_texts = list(set(all_texts))
    
    # Shuffle and limit to requested count
    random.shuffle(unique_texts)
    if len(unique_texts) > count:
        unique_texts = unique_texts[:count]
    
    logger.info(f"Created enhanced calibration dataset with {len(unique_texts)} texts")
    return unique_texts


def load_calibration_dataset(domain: str = "general", count: int = 100) -> List[str]:
    """
    Load a calibration dataset for a specific domain.
    
    This is a convenience function that provides a simplified interface
    for loading calibration datasets.
    
    Parameters
    ----------
    domain : str, default="general"
        Domain to load calibration dataset for 
        ("general", "financial", "sap", "technical", "all")
    count : int, default=100
        Number of calibration texts to load
        
    Returns
    -------
    List[str]
        Calibration dataset
    """
    if domain == "financial" and HAS_FINANCIAL_CALIBRATION:
        return get_financial_calibration_dataset(count=count)
    else:
        return get_domain_calibration_texts(domain=domain, count=count)