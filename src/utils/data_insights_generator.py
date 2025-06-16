#\!/usr/bin/env python
"""
Data Insights Generator for SAP HANA Cloud

This application uses the LangChain integration with SAP HANA Cloud to:
1. Store and retrieve financial documents
2. Generate insights from the documents using LLM
3. Track and manage insights over time
"""

import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.documents import Document

from langchain_hana_integration import SAP_HANA_VectorStore, HanaOptimizedEmbeddings
from langchain_hana_integration.connection import create_connection_pool
from langchain_hana_integration.utils.distance import DistanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("insights_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables. Some features will be limited.")


class DataInsightsGenerator:
    """Data Insights Generator for financial documents."""
    
    def __init__(
        self,
        connection_config_path: str = "config/connection.json",
        vector_table_name: str = "FINANCIAL_DOCUMENTS",
        insights_table_name: str = "FINANCIAL_INSIGHTS",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        cache_dir: str = "./cache"
    ):
        """
        Initialize the Data Insights Generator.
        
        Args:
            connection_config_path: Path to the connection configuration file
            vector_table_name: Name of the table for storing document vectors
            insights_table_name: Name of the table for storing generated insights
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the LLM model
            cache_dir: Directory for caching embeddings
        """
        self.connection_config_path = connection_config_path
        self.vector_table_name = vector_table_name
        self.insights_table_name = insights_table_name
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.cache_dir = cache_dir
        
        # Initialize components
        self._initialize_connection()
        self._initialize_vectorstore()
        self._initialize_insights_table()
        self._initialize_llm()
    
    def _initialize_connection(self):
        """Initialize connection to SAP HANA Cloud."""
        logger.info("Initializing connection to SAP HANA Cloud...")
        
        # Load connection configuration
        with open(self.connection_config_path, 'r') as f:
            connection_params = json.load(f)
        
        # Create connection pool
        create_connection_pool(
            connection_params=connection_params,
            pool_name="insights_pool",
            min_connections=1,
            max_connections=5
        )
        
        logger.info("Connection initialized successfully")
    
    def _initialize_vectorstore(self):
        """Initialize vector store for document storage and retrieval."""
        logger.info("Initializing vector store...")
        
        # Initialize embedding model
        self.embedding_model = HanaOptimizedEmbeddings(
            model_name=self.embedding_model_name,
            enable_caching=True,
            cache_dir=self.cache_dir,
            memory_cache_size=1000,
            normalize_embeddings=True
        )
        
        # Initialize vector store
        self.vector_store = SAP_HANA_VectorStore(
            embedding=self.embedding_model,
            pool_name="insights_pool",
            table_name=self.vector_table_name,
            distance_strategy=DistanceStrategy.COSINE,
            auto_create_index=True,
            batch_size=50,
            enable_logging=True
        )
        
        logger.info("Vector store initialized successfully")
    
    def _initialize_insights_table(self):
        """Initialize table for storing generated insights."""
        from langchain_hana_integration.connection import get_connection
        
        logger.info("Initializing insights table...")
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{self.insights_table_name}" (
            "DOCUMENT_ID" NVARCHAR(100),
            "INSIGHT_TEXT" NCLOB,
            "INSIGHT_TYPE" NVARCHAR(50),
            "CONFIDENCE" FLOAT,
            "TIMESTAMP" TIMESTAMP,
            "SOURCE_QUERY" NVARCHAR(500),
            "METADATA" NCLOB
        )
        """
        
        with get_connection("insights_pool") as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(create_table_sql)
                connection.commit()
                logger.info(f"Insights table '{self.insights_table_name}' initialized")
            except Exception as e:
                connection.rollback()
                logger.error(f"Error creating insights table: {e}")
                raise
            finally:
                cursor.close()
    
    def _initialize_llm(self):
        """Initialize LLM for generating insights."""
        if OPENAI_API_KEY:
            logger.info(f"Initializing LLM model: {self.llm_model_name}")
            self.llm = ChatOpenAI(
                model=self.llm_model_name,
                temperature=0,
                api_key=OPENAI_API_KEY
            )
            self.has_llm = True
        else:
            logger.warning("No OpenAI API key found. Insight generation will be disabled.")
            self.has_llm = False
    
    def add_financial_document(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a financial document to the vector store.
        
        Args:
            text: Document text
            metadata: Document metadata (company, date, type, etc.)
        """
        logger.info(f"Adding financial document: {metadata.get('title', 'Untitled')}")
        
        # Add document ID if not present
        if 'id' not in metadata:
            metadata['id'] = f"doc_{int(time.time())}_{hash(text) % 10000}"
        
        # Add document
        self.vector_store.add_texts([text], [metadata])
        
        logger.info(f"Document added with ID: {metadata['id']}")
    
    def add_financial_documents_from_csv(
        self,
        csv_path: str,
        text_column: str,
        metadata_columns: Optional[List[str]] = None
    ) -> int:
        """
        Add financial documents from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            text_column: Name of the column containing document text
            metadata_columns: Names of columns to use as metadata
            
        Returns:
            Number of documents added
        """
        logger.info(f"Adding financial documents from CSV: {csv_path}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Check if text column exists
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
            
            # Determine metadata columns
            if metadata_columns is None:
                metadata_columns = [col for col in df.columns if col \!= text_column]
            
            # Add documents
            count = 0
            for _, row in df.iterrows():
                text = row[text_column]
                
                # Skip empty documents
                if not isinstance(text, str) or not text.strip():
                    continue
                
                # Create metadata
                metadata = {col: row[col] for col in metadata_columns if col in row}
                metadata['id'] = f"csv_{int(time.time())}_{count}"
                metadata['source'] = os.path.basename(csv_path)
                
                # Add document
                self.vector_store.add_texts([text], [metadata])
                count += 1
            
            logger.info(f"Added {count} documents from CSV")
            return count
        
        except Exception as e:
            logger.error(f"Error adding documents from CSV: {e}")
            raise
    
    def search_documents(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for financial documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            filter: Optional filter
            
        Returns:
            List of matching documents
        """
        logger.info(f"Searching documents with query: '{query}'")
        
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        logger.info(f"Found {len(results)} matching documents")
        return results
    
    def generate_insight(
        self,
        query: str,
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        insight_type: str = "financial_analysis"
    ) -> Dict[str, Any]:
        """
        Generate an insight based on financial documents.
        
        Args:
            query: Specific question or analysis request
            k: Number of documents to consider
            filter: Optional filter for document selection
            insight_type: Type of insight to generate
            
        Returns:
            Generated insight with metadata
        """
        if not self.has_llm:
            raise ValueError("LLM not initialized. Cannot generate insights.")
        
        logger.info(f"Generating {insight_type} insight for query: '{query}'")
        
        # Search for relevant documents
        documents = self.search_documents(query, k, filter)
        
        if not documents:
            logger.warning("No documents found for insight generation")
            return {
                "insight": "Insufficient data to generate insight",
                "confidence": 0.0,
                "documents": [],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
        
        # Prepare context from documents
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
        
        # Define the output structure
        response_schemas = [
            ResponseSchema(name="insight", description="The main insight or analysis based on the documents"),
            ResponseSchema(name="confidence", description="Confidence score between 0.0 and 1.0"),
            ResponseSchema(name="supporting_facts", description="Key facts from the documents that support this insight"),
            ResponseSchema(name="additional_questions", description="Follow-up questions that could deepen the analysis")
        ]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        # Create prompt template
        prompt_template = """
        You are a financial analyst tasked with generating insights from financial documents.
        
        CONTEXT:
        {context}
        
        QUERY:
        {query}
        
        Based on the provided documents, generate a {insight_type} insight that addresses the query.
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "query", "insight_type"],
            partial_variables={"format_instructions": format_instructions}
        )
        
        # Generate insight
        llm_input = prompt.format(
            context=context,
            query=query,
            insight_type=insight_type
        )
        
        llm_output = self.llm.invoke(llm_input)
        
        try:
            # Parse the structured output
            parsed_output = output_parser.parse(llm_output.content)
            
            # Create result
            result = {
                "insight": parsed_output["insight"],
                "confidence": float(parsed_output["confidence"]),
                "supporting_facts": parsed_output["supporting_facts"],
                "additional_questions": parsed_output["additional_questions"],
                "documents": [{"id": doc.metadata.get("id"), "title": doc.metadata.get("title", "Untitled")} for doc in documents],
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "insight_type": insight_type
            }
            
            # Store insight in database
            self._store_insight(result, documents[0].metadata.get("id", "unknown"))
            
            logger.info(f"Generated {insight_type} insight with confidence {result['confidence']}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            logger.error(f"Raw output: {llm_output.content}")
            
            # Return a simplified result
            return {
                "insight": llm_output.content,
                "confidence": 0.5,
                "documents": [{"id": doc.metadata.get("id"), "title": doc.metadata.get("title", "Untitled")} for doc in documents],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _store_insight(self, insight: Dict[str, Any], document_id: str) -> None:
        """Store generated insight in the database."""
        from langchain_hana_integration.connection import get_connection
        
        with get_connection("insights_pool") as connection:
            cursor = connection.cursor()
            try:
                # Prepare statement
                insert_sql = f"""
                INSERT INTO "{self.insights_table_name}" (
                    "DOCUMENT_ID", "INSIGHT_TEXT", "INSIGHT_TYPE", "CONFIDENCE", 
                    "TIMESTAMP", "SOURCE_QUERY", "METADATA"
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                
                # Prepare parameters
                params = [
                    document_id,
                    insight["insight"],
                    insight["insight_type"],
                    insight["confidence"],
                    datetime.now(),
                    insight["query"],
                    json.dumps({
                        "supporting_facts": insight.get("supporting_facts", []),
                        "additional_questions": insight.get("additional_questions", []),
                        "documents": insight.get("documents", [])
                    })
                ]
                
                # Execute
                cursor.execute(insert_sql, params)
                connection.commit()
                
                logger.info(f"Stored insight for document {document_id}")
                
            except Exception as e:
                connection.rollback()
                logger.error(f"Error storing insight: {e}")
            finally:
                cursor.close()
    
    def get_insights(
        self,
        document_id: Optional[str] = None,
        insight_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored insights.
        
        Args:
            document_id: Optional document ID to filter by
            insight_type: Optional insight type to filter by
            min_confidence: Minimum confidence threshold
            limit: Maximum number of insights to return
            
        Returns:
            List of insights
        """
        from langchain_hana_integration.connection import get_connection
        
        logger.info("Retrieving insights from database")
        
        with get_connection("insights_pool") as connection:
            cursor = connection.cursor()
            try:
                # Prepare WHERE clause
                conditions = []
                params = []
                
                if document_id:
                    conditions.append('"DOCUMENT_ID" = ?')
                    params.append(document_id)
                
                if insight_type:
                    conditions.append('"INSIGHT_TYPE" = ?')
                    params.append(insight_type)
                
                if min_confidence > 0:
                    conditions.append('"CONFIDENCE" >= ?')
                    params.append(min_confidence)
                
                # Prepare SQL
                sql = f'SELECT "DOCUMENT_ID", "INSIGHT_TEXT", "INSIGHT_TYPE", "CONFIDENCE", "TIMESTAMP", "SOURCE_QUERY", "METADATA" FROM "{self.insights_table_name}"'
                
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                
                sql += f' ORDER BY "TIMESTAMP" DESC LIMIT {limit}'
                
                # Execute
                cursor.execute(sql, params)
                
                # Process results
                results = []
                for row in cursor.fetchall():
                    try:
                        metadata = json.loads(row[6]) if row[6] else {}
                    except:
                        metadata = {}
                    
                    results.append({
                        "document_id": row[0],
                        "insight": row[1],
                        "insight_type": row[2],
                        "confidence": row[3],
                        "timestamp": row[4].isoformat() if row[4] else None,
                        "query": row[5],
                        "supporting_facts": metadata.get("supporting_facts", []),
                        "additional_questions": metadata.get("additional_questions", []),
                        "documents": metadata.get("documents", [])
                    })
                
                logger.info(f"Retrieved {len(results)} insights")
                return results
                
            except Exception as e:
                logger.error(f"Error retrieving insights: {e}")
                return []
            finally:
                cursor.close()
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Returns:
            Dictionary with document statistics
        """
        from langchain_hana_integration.connection import get_connection
        
        logger.info("Retrieving document statistics")
        
        with get_connection("insights_pool") as connection:
            cursor = connection.cursor()
            try:
                # Get total document count
                cursor.execute(f'SELECT COUNT(*) FROM "{self.vector_table_name}"')
                total_count = cursor.fetchone()[0]
                
                # Get metadata statistics if available
                metadata_stats = {}
                
                # Check if table has specific metadata columns
                cursor.execute(f"""
                SELECT COLUMN_NAME FROM SYS.TABLE_COLUMNS 
                WHERE SCHEMA_NAME = CURRENT_SCHEMA 
                AND TABLE_NAME = ?
                """, (self.vector_table_name,))
                
                columns = [row[0] for row in cursor.fetchall()]
                metadata_columns = [col for col in columns if col not in [
                    self.vector_store.content_column, 
                    self.vector_store.metadata_column,
                    self.vector_store.vector_column
                ]]
                
                # Get statistics for metadata columns
                for column in metadata_columns:
                    cursor.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{self.vector_table_name}"')
                    distinct_count = cursor.fetchone()[0]
                    
                    cursor.execute(f'SELECT "{column}", COUNT(*) FROM "{self.vector_table_name}" GROUP BY "{column}" ORDER BY COUNT(*) DESC LIMIT 5')
                    top_values = [{"value": row[0], "count": row[1]} for row in cursor.fetchall()]
                    
                    metadata_stats[column] = {
                        "distinct_count": distinct_count,
                        "top_values": top_values
                    }
                
                # Get vector statistics
                vector_dimension = None
                try:
                    # Try to get dimension from a sample vector
                    cursor.execute(f'SELECT "{self.vector_store.vector_column}" FROM "{self.vector_table_name}" LIMIT 1')
                    sample_vector = cursor.fetchone()
                    if sample_vector and sample_vector[0]:
                        from langchain_hana_integration.utils.serialization import deserialize_vector
                        vector = deserialize_vector(sample_vector[0], self.vector_store.vector_column_type)
                        vector_dimension = len(vector)
                except:
                    pass
                
                # Get insight statistics
                cursor.execute(f'SELECT COUNT(*) FROM "{self.insights_table_name}"')
                insight_count = cursor.fetchone()[0]
                
                cursor.execute(f'SELECT "INSIGHT_TYPE", COUNT(*) FROM "{self.insights_table_name}" GROUP BY "INSIGHT_TYPE"')
                insight_types = [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                cursor.execute(f'SELECT AVG("CONFIDENCE") FROM "{self.insights_table_name}"')
                avg_confidence = cursor.fetchone()[0]
                
                # Build result
                result = {
                    "total_documents": total_count,
                    "metadata_statistics": metadata_stats,
                    "vector_dimension": vector_dimension,
                    "insights": {
                        "total_count": insight_count,
                        "types": insight_types,
                        "average_confidence": avg_confidence
                    },
                    "embedding_model": self.embedding_model_name,
                    "llm_model": self.llm_model_name if self.has_llm else None,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add embedding metrics if available
                if hasattr(self.embedding_model, "get_metrics"):
                    embedding_metrics = self.embedding_model.get_metrics()
                    if embedding_metrics:
                        result["embedding_metrics"] = embedding_metrics
                
                # Add vector store metrics if available
                if hasattr(self.vector_store, "get_metrics"):
                    vector_metrics = self.vector_store.get_metrics()
                    if vector_metrics:
                        result["vector_metrics"] = vector_metrics
                
                logger.info(f"Retrieved statistics for {total_count} documents")
                return result
                
            except Exception as e:
                logger.error(f"Error retrieving document statistics: {e}")
                return {"error": str(e)}
            finally:
                cursor.close()


def demo():
    """Run a demonstration of the Data Insights Generator."""
    parser = argparse.ArgumentParser(description="Data Insights Generator Demo")
    parser.add_argument("--csv", help="Path to CSV file with financial documents")
    parser.add_argument("--text-column", default="text", help="Name of the text column in the CSV")
    parser.add_argument("--query", default="What are the key financial trends?", help="Query for generating insights")
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. Set it with:")
        print("export OPENAI_API_KEY=your_api_key_here")
        print("Continuing with limited functionality...\n")
    
    print("Initializing Data Insights Generator...")
    generator = DataInsightsGenerator()
    
    # Add sample documents if no CSV provided
    if not args.csv:
        print("Adding sample financial documents...")
        sample_docs = [
            {
                "text": "Q2 2023 Financial Report: Company XYZ reported quarterly revenue of $10.2 million, up 15% year-over-year. EBITDA margin improved to a record 28.3%, compared to 24.5% in Q2 2022. Operating expenses decreased by 3% despite inflation pressures. The company's cash position strengthened to $45 million with no debt. Management raised full-year revenue guidance by 5%, citing strong demand in North American markets.",
                "metadata": {
                    "title": "Q2 2023 Financial Report",
                    "company": "XYZ Corporation",
                    "date": "2023-07-15",
                    "type": "quarterly_report",
                    "industry": "technology"
                }
            },
            {
                "text": "Market Analysis Report: The fintech sector is experiencing a significant transformation, with digital payment solutions growing at 25% annually. Regulatory changes in Europe are expected to increase compliance costs by an estimated 12-15% for smaller players. Venture capital investments in fintech declined by 30% in H1 2023 compared to the same period last year. Market consolidation is accelerating, with large financial institutions acquiring technology startups to enhance their digital capabilities.",
                "metadata": {
                    "title": "Fintech Market Analysis",
                    "company": "Financial Research Institute",
                    "date": "2023-08-10",
                    "type": "market_analysis",
                    "industry": "financial_services"
                }
            },
            {
                "text": "Investor Presentation: Our company achieved a 22% return on equity in 2022, significantly outperforming the industry average of 15%. Capital expenditures are planned at $25-30 million for 2023, focused on digital infrastructure and automation. The company's debt-to-equity ratio remains healthy at 0.4, providing ample flexibility for strategic acquisitions. Management expects 10-12% compound annual growth rate over the next three years, driven by expansion into Asian markets and new product lines.",
                "metadata": {
                    "title": "2023 Investor Presentation",
                    "company": "ABC Financial",
                    "date": "2023-03-22",
                    "type": "investor_presentation",
                    "industry": "financial_services"
                }
            },
            {
                "text": "Industry Risk Assessment: The manufacturing sector faces increasing margin pressure due to rising raw material costs, up 18% since January. Supply chain disruptions continue to affect production schedules, with average lead times extending from 30 to 45 days. Energy cost volatility presents a significant challenge, with costs up 23% year-over-year in Q1 2023. Companies with robust hedging strategies have outperformed peers by an average of 7% in terms of profitability. Market leaders are accelerating investments in automation to mitigate labor cost pressures.",
                "metadata": {
                    "title": "Manufacturing Sector Risk Assessment",
                    "company": "Industrial Analytics Group",
                    "date": "2023-05-05",
                    "type": "risk_assessment",
                    "industry": "manufacturing"
                }
            },
            {
                "text": "Economic Outlook: Interest rates are projected to remain elevated through 2023, with consensus forecasts indicating one more 25 basis point increase. Inflation is showing signs of moderation, with core CPI expected to decline to 3.1% by year-end. Consumer spending remains resilient despite reduced purchasing power, with retail sales up 2.3% year-over-year in real terms. Corporate earnings growth is expected to slow to 4-6% in 2023, down from 12% in 2022. The labor market is gradually cooling, with job openings declining 15% from their peak but still above pre-pandemic levels.",
                "metadata": {
                    "title": "Q3 2023 Economic Outlook",
                    "company": "Economic Research Institute",
                    "date": "2023-07-28",
                    "type": "economic_forecast",
                    "industry": "all"
                }
            }
        ]
        
        for doc in sample_docs:
            generator.add_financial_document(doc["text"], doc["metadata"])
        
        print(f"Added {len(sample_docs)} sample documents\n")
    else:
        # Add documents from CSV
        if not os.path.exists(args.csv):
            print(f"Error: CSV file '{args.csv}' not found")
            return
        
        print(f"Adding documents from CSV file: {args.csv}")
        count = generator.add_financial_documents_from_csv(args.csv, args.text_column)
        print(f"Added {count} documents from CSV\n")
    
    # Get document statistics
    print("Document Statistics:")
    stats = generator.get_document_statistics()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Embedding model: {stats['embedding_model']}")
    print(f"LLM model: {stats.get('llm_model', 'Not available')}\n")
    
    # Search for documents
    print(f"Searching for documents with query: '{args.query}'")
    search_results = generator.search_documents(args.query, k=3)
    
    print(f"Found {len(search_results)} relevant documents:")
    for i, doc in enumerate(search_results):
        print(f"\nDocument {i+1}:")
        print(f"Title: {doc.metadata.get('title', 'Untitled')}")
        print(f"Company: {doc.metadata.get('company', 'Unknown')}")
        print(f"Date: {doc.metadata.get('date', 'Unknown')}")
        print(f"Type: {doc.metadata.get('type', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
    
    # Generate insights if OpenAI API key is available
    if OPENAI_API_KEY:
        print("\nGenerating financial insights...")
        insight = generator.generate_insight(args.query)
        
        print("\nGenerated Insight:")
        print(f"Query: {insight['query']}")
        print(f"Insight: {insight['insight']}")
        print(f"Confidence: {insight['confidence']}")
        
        print("\nSupporting Facts:")
        for fact in insight.get('supporting_facts', []):
            print(f"- {fact}")
        
        print("\nAdditional Questions:")
        for question in insight.get('additional_questions', []):
            print(f"- {question}")
    else:
        print("\nSkipping insight generation (OPENAI_API_KEY not set)")
    
    print("\nDemo completed successfully\!")


if __name__ == "__main__":
    demo()
EOF < /dev/null