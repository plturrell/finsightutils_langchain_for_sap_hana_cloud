"""
User feedback collection module for improving vector transformations.

This module provides tools for collecting, storing, and applying user feedback
on vector transformations to continuously improve the system.
"""

import uuid
import time
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from abc import ABC, abstractmethod

import numpy as np
from hdbcli import dbapi

from langchain_hana.reasoning.data_lineage import LineageTracker, LineageEvent
from langchain_hana.reasoning.transparent_pipeline import TransparentEmbeddingPipeline

logger = logging.getLogger(__name__)


class FeedbackItem:
    """
    Represents a single piece of user feedback.
    
    Captures user feedback on a specific aspect of the system.
    """
    
    def __init__(
        self,
        feedback_id: str,
        feedback_type: str,
        content: Dict[str, Any],
        user_id: Optional[str] = None,
        target_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a feedback item.
        
        Args:
            feedback_id: Unique identifier for this feedback
            feedback_type: Type of feedback (e.g., 'embedding', 'retrieval', 'reasoning')
            content: Content of the feedback
            user_id: Optional identifier for the user providing feedback
            target_id: Optional identifier for the target of the feedback
            timestamp: Time when this feedback was created
            metadata: Additional metadata about the feedback
        """
        self.feedback_id = feedback_id
        self.feedback_type = feedback_type
        self.content = content
        self.user_id = user_id
        self.target_id = target_id
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the feedback item to a dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type,
            "content": self.content,
            "user_id": self.user_id,
            "target_id": self.target_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackItem":
        """Create a feedback item from a dictionary."""
        return cls(
            feedback_id=data["feedback_id"],
            feedback_type=data["feedback_type"],
            content=data["content"],
            user_id=data.get("user_id"),
            target_id=data.get("target_id"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class EmbeddingFeedback(FeedbackItem):
    """
    Feedback specific to embedding quality.
    
    Captures user feedback on the quality of embeddings and suggestions for improvement.
    """
    
    def __init__(
        self,
        rating: int,
        text: str,
        embedding_id: str,
        suggestions: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize embedding feedback.
        
        Args:
            rating: User rating (1-5)
            text: The text that was embedded
            embedding_id: ID of the embedding
            suggestions: Optional suggestions for improvement
            user_id: Optional identifier for the user providing feedback
            timestamp: Time when this feedback was created
            metadata: Additional metadata about the feedback
        """
        content = {
            "rating": rating,
            "text": text,
            "suggestions": suggestions or [],
        }
        
        super().__init__(
            feedback_id=str(uuid.uuid4()),
            feedback_type="embedding",
            content=content,
            user_id=user_id,
            target_id=embedding_id,
            timestamp=timestamp,
            metadata=metadata,
        )


class RetrievalFeedback(FeedbackItem):
    """
    Feedback specific to retrieval quality.
    
    Captures user feedback on the quality of retrieval results and relevance judgments.
    """
    
    def __init__(
        self,
        query: str,
        relevant_results: List[str],
        irrelevant_results: List[str],
        query_id: str,
        missing_results: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize retrieval feedback.
        
        Args:
            query: The query that was searched
            relevant_results: IDs of results deemed relevant
            irrelevant_results: IDs of results deemed irrelevant
            query_id: ID of the query
            missing_results: Optional IDs of results that should have been included
            user_id: Optional identifier for the user providing feedback
            timestamp: Time when this feedback was created
            metadata: Additional metadata about the feedback
        """
        content = {
            "query": query,
            "relevant_results": relevant_results,
            "irrelevant_results": irrelevant_results,
            "missing_results": missing_results or [],
        }
        
        super().__init__(
            feedback_id=str(uuid.uuid4()),
            feedback_type="retrieval",
            content=content,
            user_id=user_id,
            target_id=query_id,
            timestamp=timestamp,
            metadata=metadata,
        )


class ReasoningFeedback(FeedbackItem):
    """
    Feedback specific to reasoning quality.
    
    Captures user feedback on the quality of reasoning steps and corrections.
    """
    
    def __init__(
        self,
        rating: int,
        reasoning_id: str,
        corrections: Optional[List[Dict[str, Any]]] = None,
        explanation: Optional[str] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize reasoning feedback.
        
        Args:
            rating: User rating (1-5)
            reasoning_id: ID of the reasoning process
            corrections: Optional corrections to reasoning steps
            explanation: Optional explanation of the feedback
            user_id: Optional identifier for the user providing feedback
            timestamp: Time when this feedback was created
            metadata: Additional metadata about the feedback
        """
        content = {
            "rating": rating,
            "corrections": corrections or [],
            "explanation": explanation or "",
        }
        
        super().__init__(
            feedback_id=str(uuid.uuid4()),
            feedback_type="reasoning",
            content=content,
            user_id=user_id,
            target_id=reasoning_id,
            timestamp=timestamp,
            metadata=metadata,
        )


class TransformationFeedback(FeedbackItem):
    """
    Feedback specific to vector transformation quality.
    
    Captures user feedback on the quality of vector transformations and suggestions.
    """
    
    def __init__(
        self,
        rating: int,
        stage: str,
        transformation_id: str,
        suggestions: Optional[List[str]] = None,
        example_inputs: Optional[List[Any]] = None,
        example_expected_outputs: Optional[List[Any]] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize transformation feedback.
        
        Args:
            rating: User rating (1-5)
            stage: The pipeline stage this feedback relates to
            transformation_id: ID of the transformation
            suggestions: Optional suggestions for improvement
            example_inputs: Optional example inputs for the transformation
            example_expected_outputs: Optional expected outputs for the example inputs
            user_id: Optional identifier for the user providing feedback
            timestamp: Time when this feedback was created
            metadata: Additional metadata about the feedback
        """
        content = {
            "rating": rating,
            "stage": stage,
            "suggestions": suggestions or [],
            "example_inputs": example_inputs or [],
            "example_expected_outputs": example_expected_outputs or [],
        }
        
        super().__init__(
            feedback_id=str(uuid.uuid4()),
            feedback_type="transformation",
            content=content,
            user_id=user_id,
            target_id=transformation_id,
            timestamp=timestamp,
            metadata=metadata,
        )


class FeedbackStorage(ABC):
    """
    Abstract base class for feedback storage backends.
    
    Defines the interface for storing and retrieving feedback.
    """
    
    @abstractmethod
    def save_feedback(self, feedback_id: str, feedback_data: Dict[str, Any]) -> bool:
        """
        Save feedback data.
        
        Args:
            feedback_id: ID of the feedback
            feedback_data: Feedback data to save
            
        Returns:
            True if the feedback was saved successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def load_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Load feedback data.
        
        Args:
            feedback_id: ID of the feedback to load
            
        Returns:
            Feedback data if found, None otherwise
        """
        pass
    
    @abstractmethod
    def load_feedback_by_target(
        self, target_id: str, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load feedback data by target ID.
        
        Args:
            target_id: Target ID to filter by
            feedback_type: Optional feedback type to filter by
            
        Returns:
            List of feedback data items for the specified target
        """
        pass
    
    @abstractmethod
    def load_feedback_by_user(
        self, user_id: str, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load feedback data by user ID.
        
        Args:
            user_id: User ID to filter by
            feedback_type: Optional feedback type to filter by
            
        Returns:
            List of feedback data items from the specified user
        """
        pass
    
    @abstractmethod
    def delete_feedback(self, feedback_id: str) -> bool:
        """
        Delete feedback data.
        
        Args:
            feedback_id: ID of the feedback to delete
            
        Returns:
            True if the feedback was deleted successfully, False otherwise
        """
        pass


class InMemoryFeedbackStorage(FeedbackStorage):
    """
    In-memory implementation of feedback storage.
    
    Stores feedback in memory, suitable for testing and development.
    """
    
    def __init__(self):
        """Initialize in-memory feedback storage."""
        self.feedback_data = {}
        self.target_index = {}
        self.user_index = {}
    
    def save_feedback(self, feedback_id: str, feedback_data: Dict[str, Any]) -> bool:
        """
        Save feedback data in memory.
        
        Args:
            feedback_id: ID of the feedback
            feedback_data: Feedback data to save
            
        Returns:
            True if the feedback was saved successfully
        """
        self.feedback_data[feedback_id] = feedback_data
        
        # Update indices
        target_id = feedback_data.get("target_id")
        if target_id:
            if target_id not in self.target_index:
                self.target_index[target_id] = set()
            self.target_index[target_id].add(feedback_id)
        
        user_id = feedback_data.get("user_id")
        if user_id:
            if user_id not in self.user_index:
                self.user_index[user_id] = set()
            self.user_index[user_id].add(feedback_id)
        
        return True
    
    def load_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Load feedback data from memory.
        
        Args:
            feedback_id: ID of the feedback to load
            
        Returns:
            Feedback data if found, None otherwise
        """
        return self.feedback_data.get(feedback_id)
    
    def load_feedback_by_target(
        self, target_id: str, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load feedback data by target ID from memory.
        
        Args:
            target_id: Target ID to filter by
            feedback_type: Optional feedback type to filter by
            
        Returns:
            List of feedback data items for the specified target
        """
        result = []
        
        feedback_ids = self.target_index.get(target_id, set())
        for feedback_id in feedback_ids:
            feedback_data = self.feedback_data.get(feedback_id)
            if feedback_data and (feedback_type is None or feedback_data.get("feedback_type") == feedback_type):
                result.append(feedback_data)
        
        return result
    
    def load_feedback_by_user(
        self, user_id: str, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load feedback data by user ID from memory.
        
        Args:
            user_id: User ID to filter by
            feedback_type: Optional feedback type to filter by
            
        Returns:
            List of feedback data items from the specified user
        """
        result = []
        
        feedback_ids = self.user_index.get(user_id, set())
        for feedback_id in feedback_ids:
            feedback_data = self.feedback_data.get(feedback_id)
            if feedback_data and (feedback_type is None or feedback_data.get("feedback_type") == feedback_type):
                result.append(feedback_data)
        
        return result
    
    def delete_feedback(self, feedback_id: str) -> bool:
        """
        Delete feedback data from memory.
        
        Args:
            feedback_id: ID of the feedback to delete
            
        Returns:
            True if the feedback was deleted successfully, False otherwise
        """
        if feedback_id not in self.feedback_data:
            return False
        
        feedback_data = self.feedback_data[feedback_id]
        
        # Update indices
        target_id = feedback_data.get("target_id")
        if target_id and target_id in self.target_index:
            self.target_index[target_id].discard(feedback_id)
            if not self.target_index[target_id]:
                del self.target_index[target_id]
        
        user_id = feedback_data.get("user_id")
        if user_id and user_id in self.user_index:
            self.user_index[user_id].discard(feedback_id)
            if not self.user_index[user_id]:
                del self.user_index[user_id]
        
        # Remove the feedback data
        del self.feedback_data[feedback_id]
        
        return True


class HanaFeedbackStorage(FeedbackStorage):
    """
    SAP HANA implementation of feedback storage.
    
    Stores feedback in a SAP HANA database table.
    """
    
    def __init__(
        self,
        connection: dbapi.Connection,
        schema_name: Optional[str] = None,
        table_name: str = "FEEDBACK_ITEMS",
    ):
        """
        Initialize HANA feedback storage.
        
        Args:
            connection: SAP HANA database connection
            schema_name: Database schema to use
            table_name: Name of the table to store feedback
        """
        self.connection = connection
        self.schema_name = schema_name
        self.table_name = table_name
        
        # Initialize feedback table
        self._initialize_feedback_table()
    
    def _initialize_feedback_table(self) -> None:
        """Initialize the feedback table in the database."""
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Create schema if it doesn't exist
            if self.schema_name:
                try:
                    cursor.execute(f"CREATE SCHEMA {self.schema_name}")
                    logger.info(f"Created schema {self.schema_name}")
                except Exception as e:
                    # Ignore error if schema already exists
                    if "exists" not in str(e).lower():
                        logger.warning(f"Error creating schema: {str(e)}")
            
            # Create feedback table if it doesn't exist
            table_sql = f"""
            CREATE TABLE IF NOT EXISTS {full_table_name} (
                FEEDBACK_ID VARCHAR(100) PRIMARY KEY,
                FEEDBACK_TYPE VARCHAR(100),
                USER_ID VARCHAR(100),
                TARGET_ID VARCHAR(100),
                TIMESTAMP TIMESTAMP,
                CONTENT NCLOB,
                METADATA NCLOB
            )
            """
            cursor.execute(table_sql)
            
            # Create indices
            try:
                cursor.execute(f"CREATE INDEX IDX_{self.table_name}_TYPE ON {full_table_name}(FEEDBACK_TYPE)")
            except Exception as e:
                # Ignore error if index already exists
                if "exists" not in str(e).lower():
                    logger.warning(f"Error creating index: {str(e)}")
            
            try:
                cursor.execute(f"CREATE INDEX IDX_{self.table_name}_TARGET ON {full_table_name}(TARGET_ID)")
            except Exception as e:
                # Ignore error if index already exists
                if "exists" not in str(e).lower():
                    logger.warning(f"Error creating index: {str(e)}")
            
            try:
                cursor.execute(f"CREATE INDEX IDX_{self.table_name}_USER ON {full_table_name}(USER_ID)")
            except Exception as e:
                # Ignore error if index already exists
                if "exists" not in str(e).lower():
                    logger.warning(f"Error creating index: {str(e)}")
            
            logger.info(f"Initialized feedback table {full_table_name}")
        except Exception as e:
            logger.error(f"Error initializing feedback table: {str(e)}")
        finally:
            cursor.close()
    
    def save_feedback(self, feedback_id: str, feedback_data: Dict[str, Any]) -> bool:
        """
        Save feedback data to HANA.
        
        Args:
            feedback_id: ID of the feedback
            feedback_data: Feedback data to save
            
        Returns:
            True if the feedback was saved successfully, False otherwise
        """
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Extract fields
            feedback_type = feedback_data.get("feedback_type", "")
            user_id = feedback_data.get("user_id")
            target_id = feedback_data.get("target_id")
            timestamp = feedback_data.get("timestamp", time.time())
            content = json.dumps(feedback_data.get("content", {}))
            metadata = json.dumps(feedback_data.get("metadata", {}))
            
            # Insert or update the feedback
            upsert_sql = f"""
            MERGE INTO {full_table_name} AS target
            USING (SELECT ? AS FEEDBACK_ID) AS source
            ON target.FEEDBACK_ID = source.FEEDBACK_ID
            WHEN MATCHED THEN
                UPDATE SET
                    FEEDBACK_TYPE = ?,
                    USER_ID = ?,
                    TARGET_ID = ?,
                    TIMESTAMP = ?,
                    CONTENT = ?,
                    METADATA = ?
            WHEN NOT MATCHED THEN
                INSERT (FEEDBACK_ID, FEEDBACK_TYPE, USER_ID, TARGET_ID, TIMESTAMP, CONTENT, METADATA)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(
                upsert_sql,
                (
                    feedback_id,
                    feedback_type,
                    user_id,
                    target_id,
                    timestamp,
                    content,
                    metadata,
                    feedback_id,
                    feedback_type,
                    user_id,
                    target_id,
                    timestamp,
                    content,
                    metadata,
                )
            )
            
            # Commit the transaction
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            try:
                self.connection.rollback()
            except:
                pass
            return False
        finally:
            cursor.close()
    
    def load_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Load feedback data from HANA.
        
        Args:
            feedback_id: ID of the feedback to load
            
        Returns:
            Feedback data if found, None otherwise
        """
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Query the feedback
            query_sql = f"""
            SELECT FEEDBACK_ID, FEEDBACK_TYPE, USER_ID, TARGET_ID, TIMESTAMP, CONTENT, METADATA
            FROM {full_table_name}
            WHERE FEEDBACK_ID = ?
            """
            
            cursor.execute(query_sql, (feedback_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse fields
            feedback_id, feedback_type, user_id, target_id, timestamp = row[0:5]
            content_json, metadata_json = row[5:7]
            
            # Parse JSON fields
            try:
                content = json.loads(content_json)
                metadata = json.loads(metadata_json)
            except:
                # If JSON parsing fails, use empty dictionaries
                content, metadata = {}, {}
            
            # Build feedback data
            feedback_data = {
                "feedback_id": feedback_id,
                "feedback_type": feedback_type,
                "user_id": user_id,
                "target_id": target_id,
                "timestamp": timestamp,
                "content": content,
                "metadata": metadata,
            }
            
            return feedback_data
        except Exception as e:
            logger.error(f"Error loading feedback: {str(e)}")
            return None
        finally:
            cursor.close()
    
    def load_feedback_by_target(
        self, target_id: str, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load feedback data by target ID from HANA.
        
        Args:
            target_id: Target ID to filter by
            feedback_type: Optional feedback type to filter by
            
        Returns:
            List of feedback data items for the specified target
        """
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Build query
            if feedback_type:
                query_sql = f"""
                SELECT FEEDBACK_ID, FEEDBACK_TYPE, USER_ID, TARGET_ID, TIMESTAMP, CONTENT, METADATA
                FROM {full_table_name}
                WHERE TARGET_ID = ? AND FEEDBACK_TYPE = ?
                ORDER BY TIMESTAMP DESC
                """
                cursor.execute(query_sql, (target_id, feedback_type))
            else:
                query_sql = f"""
                SELECT FEEDBACK_ID, FEEDBACK_TYPE, USER_ID, TARGET_ID, TIMESTAMP, CONTENT, METADATA
                FROM {full_table_name}
                WHERE TARGET_ID = ?
                ORDER BY TIMESTAMP DESC
                """
                cursor.execute(query_sql, (target_id,))
            
            result = []
            
            # Process results
            for row in cursor.fetchall():
                feedback_id, feedback_type, user_id, target_id, timestamp = row[0:5]
                content_json, metadata_json = row[5:7]
                
                # Parse JSON fields
                try:
                    content = json.loads(content_json)
                    metadata = json.loads(metadata_json)
                except:
                    # If JSON parsing fails, use empty dictionaries
                    content, metadata = {}, {}
                
                # Build feedback data
                feedback_data = {
                    "feedback_id": feedback_id,
                    "feedback_type": feedback_type,
                    "user_id": user_id,
                    "target_id": target_id,
                    "timestamp": timestamp,
                    "content": content,
                    "metadata": metadata,
                }
                
                result.append(feedback_data)
            
            return result
        except Exception as e:
            logger.error(f"Error loading feedback by target: {str(e)}")
            return []
        finally:
            cursor.close()
    
    def load_feedback_by_user(
        self, user_id: str, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load feedback data by user ID from HANA.
        
        Args:
            user_id: User ID to filter by
            feedback_type: Optional feedback type to filter by
            
        Returns:
            List of feedback data items from the specified user
        """
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Build query
            if feedback_type:
                query_sql = f"""
                SELECT FEEDBACK_ID, FEEDBACK_TYPE, USER_ID, TARGET_ID, TIMESTAMP, CONTENT, METADATA
                FROM {full_table_name}
                WHERE USER_ID = ? AND FEEDBACK_TYPE = ?
                ORDER BY TIMESTAMP DESC
                """
                cursor.execute(query_sql, (user_id, feedback_type))
            else:
                query_sql = f"""
                SELECT FEEDBACK_ID, FEEDBACK_TYPE, USER_ID, TARGET_ID, TIMESTAMP, CONTENT, METADATA
                FROM {full_table_name}
                WHERE USER_ID = ?
                ORDER BY TIMESTAMP DESC
                """
                cursor.execute(query_sql, (user_id,))
            
            result = []
            
            # Process results
            for row in cursor.fetchall():
                feedback_id, feedback_type, user_id, target_id, timestamp = row[0:5]
                content_json, metadata_json = row[5:7]
                
                # Parse JSON fields
                try:
                    content = json.loads(content_json)
                    metadata = json.loads(metadata_json)
                except:
                    # If JSON parsing fails, use empty dictionaries
                    content, metadata = {}, {}
                
                # Build feedback data
                feedback_data = {
                    "feedback_id": feedback_id,
                    "feedback_type": feedback_type,
                    "user_id": user_id,
                    "target_id": target_id,
                    "timestamp": timestamp,
                    "content": content,
                    "metadata": metadata,
                }
                
                result.append(feedback_data)
            
            return result
        except Exception as e:
            logger.error(f"Error loading feedback by user: {str(e)}")
            return []
        finally:
            cursor.close()
    
    def delete_feedback(self, feedback_id: str) -> bool:
        """
        Delete feedback data from HANA.
        
        Args:
            feedback_id: ID of the feedback to delete
            
        Returns:
            True if the feedback was deleted successfully, False otherwise
        """
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Delete the feedback
            delete_sql = f"""
            DELETE FROM {full_table_name}
            WHERE FEEDBACK_ID = ?
            """
            
            cursor.execute(delete_sql, (feedback_id,))
            
            # Commit the transaction
            self.connection.commit()
            
            # Check if any rows were affected
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting feedback: {str(e)}")
            try:
                self.connection.rollback()
            except:
                pass
            return False
        finally:
            cursor.close()


class FeedbackProcessor:
    """
    Processes feedback to improve system performance.
    
    Analyzes feedback and applies insights to improve embeddings, retrieval, and reasoning.
    """
    
    def __init__(self, pipeline: Optional[TransparentEmbeddingPipeline] = None):
        """
        Initialize a feedback processor.
        
        Args:
            pipeline: Optional embedding pipeline to improve
        """
        self.pipeline = pipeline
        self.relevance_judgments = {}
        self.embedding_feedback = {}
        self.reasoning_feedback = {}
        self.transformation_feedback = {}
    
    def process_embedding_feedback(self, feedback_item: FeedbackItem) -> None:
        """
        Process embedding feedback.
        
        Args:
            feedback_item: Feedback item to process
        """
        if feedback_item.feedback_type != "embedding":
            logger.warning(f"Expected embedding feedback, got {feedback_item.feedback_type}")
            return
        
        # Extract data
        target_id = feedback_item.target_id
        if not target_id:
            logger.warning("Embedding feedback missing target_id")
            return
        
        content = feedback_item.content
        rating = content.get("rating")
        text = content.get("text")
        suggestions = content.get("suggestions", [])
        
        if rating is None or text is None:
            logger.warning("Embedding feedback missing rating or text")
            return
        
        # Store feedback for this embedding
        if target_id not in self.embedding_feedback:
            self.embedding_feedback[target_id] = []
        self.embedding_feedback[target_id].append(feedback_item)
        
        # Log processing
        logger.info(f"Processed embedding feedback for {target_id}: rating={rating}, suggestions={len(suggestions)}")
        
        # Apply feedback to improve embeddings
        # In a real implementation, this would use the feedback to fine-tune embeddings
        if self.pipeline and rating < 3:
            logger.info(f"Low embedding rating ({rating}) detected for '{text[:50]}...'")
            # This is a placeholder for feedback-based improvement
    
    def process_retrieval_feedback(self, feedback_item: FeedbackItem) -> None:
        """
        Process retrieval feedback.
        
        Args:
            feedback_item: Feedback item to process
        """
        if feedback_item.feedback_type != "retrieval":
            logger.warning(f"Expected retrieval feedback, got {feedback_item.feedback_type}")
            return
        
        # Extract data
        target_id = feedback_item.target_id
        if not target_id:
            logger.warning("Retrieval feedback missing target_id")
            return
        
        content = feedback_item.content
        query = content.get("query")
        relevant_results = content.get("relevant_results", [])
        irrelevant_results = content.get("irrelevant_results", [])
        missing_results = content.get("missing_results", [])
        
        if query is None:
            logger.warning("Retrieval feedback missing query")
            return
        
        # Store relevance judgments
        if query not in self.relevance_judgments:
            self.relevance_judgments[query] = {
                "relevant": set(),
                "irrelevant": set(),
                "missing": set(),
            }
        
        for result_id in relevant_results:
            self.relevance_judgments[query]["relevant"].add(result_id)
            self.relevance_judgments[query]["irrelevant"].discard(result_id)
        
        for result_id in irrelevant_results:
            self.relevance_judgments[query]["irrelevant"].add(result_id)
            self.relevance_judgments[query]["relevant"].discard(result_id)
        
        for result_id in missing_results:
            self.relevance_judgments[query]["missing"].add(result_id)
        
        # Log processing
        logger.info(f"Processed retrieval feedback for {target_id}: query='{query[:50]}...', relevant={len(relevant_results)}, irrelevant={len(irrelevant_results)}, missing={len(missing_results)}")
        
        # Apply feedback to improve retrieval
        # In a real implementation, this would use the feedback to fine-tune retrieval
    
    def process_reasoning_feedback(self, feedback_item: FeedbackItem) -> None:
        """
        Process reasoning feedback.
        
        Args:
            feedback_item: Feedback item to process
        """
        if feedback_item.feedback_type != "reasoning":
            logger.warning(f"Expected reasoning feedback, got {feedback_item.feedback_type}")
            return
        
        # Extract data
        target_id = feedback_item.target_id
        if not target_id:
            logger.warning("Reasoning feedback missing target_id")
            return
        
        content = feedback_item.content
        rating = content.get("rating")
        corrections = content.get("corrections", [])
        explanation = content.get("explanation", "")
        
        if rating is None:
            logger.warning("Reasoning feedback missing rating")
            return
        
        # Store feedback for this reasoning
        if target_id not in self.reasoning_feedback:
            self.reasoning_feedback[target_id] = []
        self.reasoning_feedback[target_id].append(feedback_item)
        
        # Log processing
        logger.info(f"Processed reasoning feedback for {target_id}: rating={rating}, corrections={len(corrections)}")
        
        # Apply feedback to improve reasoning
        # In a real implementation, this would use the feedback to fine-tune reasoning
    
    def process_transformation_feedback(self, feedback_item: FeedbackItem) -> None:
        """
        Process transformation feedback.
        
        Args:
            feedback_item: Feedback item to process
        """
        if feedback_item.feedback_type != "transformation":
            logger.warning(f"Expected transformation feedback, got {feedback_item.feedback_type}")
            return
        
        # Extract data
        target_id = feedback_item.target_id
        if not target_id:
            logger.warning("Transformation feedback missing target_id")
            return
        
        content = feedback_item.content
        rating = content.get("rating")
        stage = content.get("stage")
        suggestions = content.get("suggestions", [])
        example_inputs = content.get("example_inputs", [])
        example_expected_outputs = content.get("example_expected_outputs", [])
        
        if rating is None or stage is None:
            logger.warning("Transformation feedback missing rating or stage")
            return
        
        # Store feedback for this transformation
        if target_id not in self.transformation_feedback:
            self.transformation_feedback[target_id] = []
        self.transformation_feedback[target_id].append(feedback_item)
        
        # Log processing
        logger.info(f"Processed transformation feedback for {target_id}, stage={stage}: rating={rating}, suggestions={len(suggestions)}, examples={len(example_inputs)}")
        
        # Apply feedback to improve transformations
        # In a real implementation, this would use the feedback to fine-tune transformations
        if self.pipeline and rating < 3 and stage in ["preprocessing", "embedding", "postprocessing"]:
            logger.info(f"Low transformation rating ({rating}) detected for stage '{stage}'")
            # This is a placeholder for feedback-based improvement
    
    def get_retrieval_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate retrieval performance metrics based on feedback.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "queries": 0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "by_query": {},
        }
        
        if not self.relevance_judgments:
            return metrics
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        
        for query, judgments in self.relevance_judgments.items():
            relevant = judgments["relevant"]
            irrelevant = judgments["irrelevant"]
            missing = judgments["missing"]
            
            # Skip queries with no judgments
            if not relevant and not irrelevant and not missing:
                continue
            
            # Calculate metrics
            # In a real implementation, these would be calculated based on actual retrieval results
            # For now, we use a simplified calculation based on judgments
            retrieved = relevant.union(irrelevant)
            true_relevant = relevant.union(missing)
            
            if not retrieved:
                precision = 0.0
            else:
                precision = len(relevant) / len(retrieved)
            
            if not true_relevant:
                recall = 1.0
            else:
                recall = len(relevant) / len(true_relevant)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            # Add to totals
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            
            # Store per-query metrics
            metrics["by_query"][query] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "relevant_count": len(relevant),
                "irrelevant_count": len(irrelevant),
                "missing_count": len(missing),
            }
        
        # Calculate averages
        query_count = len(metrics["by_query"])
        metrics["queries"] = query_count
        
        if query_count > 0:
            metrics["avg_precision"] = total_precision / query_count
            metrics["avg_recall"] = total_recall / query_count
            metrics["avg_f1"] = total_f1 / query_count
        
        return metrics
    
    def get_embedding_quality_metrics(self) -> Dict[str, Any]:
        """
        Calculate embedding quality metrics based on feedback.
        
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "embeddings": 0,
            "avg_rating": 0.0,
            "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "suggestion_count": 0,
        }
        
        if not self.embedding_feedback:
            return metrics
        
        total_rating = 0.0
        total_suggestions = 0
        
        for target_id, feedback_items in self.embedding_feedback.items():
            for item in feedback_items:
                rating = item.content.get("rating")
                suggestions = item.content.get("suggestions", [])
                
                if rating:
                    total_rating += rating
                    metrics["rating_distribution"][rating] = metrics["rating_distribution"].get(rating, 0) + 1
                
                total_suggestions += len(suggestions)
        
        # Calculate totals
        feedback_count = sum(len(items) for items in self.embedding_feedback.values())
        metrics["embeddings"] = len(self.embedding_feedback)
        metrics["suggestion_count"] = total_suggestions
        
        if feedback_count > 0:
            metrics["avg_rating"] = total_rating / feedback_count
        
        return metrics
    
    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations for system improvements based on feedback.
        
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Check embedding quality
        embedding_metrics = self.get_embedding_quality_metrics()
        if embedding_metrics["avg_rating"] < 3.5 and embedding_metrics["embeddings"] > 5:
            recommendations.append({
                "type": "embedding",
                "priority": "high" if embedding_metrics["avg_rating"] < 2.5 else "medium",
                "description": f"Improve embedding quality (current avg rating: {embedding_metrics['avg_rating']:.2f})",
                "evidence": f"Based on {embedding_metrics['embeddings']} embeddings with user feedback",
            })
        
        # Check retrieval performance
        retrieval_metrics = self.get_retrieval_performance_metrics()
        if retrieval_metrics["avg_f1"] < 0.7 and retrieval_metrics["queries"] > 5:
            recommendations.append({
                "type": "retrieval",
                "priority": "high" if retrieval_metrics["avg_f1"] < 0.5 else "medium",
                "description": f"Improve retrieval accuracy (current F1: {retrieval_metrics['avg_f1']:.2f})",
                "evidence": f"Based on {retrieval_metrics['queries']} queries with user feedback",
            })
        
        # Check reasoning quality
        reasoning_count = len(self.reasoning_feedback)
        if reasoning_count > 0:
            low_rating_count = 0
            for items in self.reasoning_feedback.values():
                for item in items:
                    if item.content.get("rating", 0) < 3:
                        low_rating_count += 1
            
            if low_rating_count > reasoning_count / 3:
                recommendations.append({
                    "type": "reasoning",
                    "priority": "high" if low_rating_count > reasoning_count / 2 else "medium",
                    "description": "Improve reasoning quality",
                    "evidence": f"{low_rating_count} out of {reasoning_count} reasoning instances received low ratings",
                })
        
        # Check transformation quality
        transformation_count = len(self.transformation_feedback)
        if transformation_count > 0:
            stage_ratings = {}
            
            for items in self.transformation_feedback.values():
                for item in items:
                    stage = item.content.get("stage", "unknown")
                    rating = item.content.get("rating", 0)
                    
                    if stage not in stage_ratings:
                        stage_ratings[stage] = {"total": 0, "sum": 0}
                    
                    stage_ratings[stage]["total"] += 1
                    stage_ratings[stage]["sum"] += rating
            
            for stage, data in stage_ratings.items():
                if data["total"] > 0:
                    avg_rating = data["sum"] / data["total"]
                    if avg_rating < 3:
                        recommendations.append({
                            "type": "transformation",
                            "priority": "high" if avg_rating < 2 else "medium",
                            "description": f"Improve '{stage}' transformation stage",
                            "evidence": f"Average rating: {avg_rating:.2f} from {data['total']} feedback items",
                        })
        
        return recommendations


class UserFeedbackCollector:
    """
    Collects and processes user feedback.
    
    Provides tools for collecting, storing, and applying user feedback to
    continuously improve the system.
    """
    
    def __init__(
        self, 
        storage_backend: Optional[FeedbackStorage] = None,
        lineage_tracker: Optional[LineageTracker] = None,
        pipeline: Optional[TransparentEmbeddingPipeline] = None,
    ):
        """
        Initialize a user feedback collector.
        
        Args:
            storage_backend: Optional backend for storing feedback
            lineage_tracker: Optional lineage tracker for tracking data transformations
            pipeline: Optional embedding pipeline to improve
        """
        self.feedback_items = {}
        self.storage_backend = storage_backend or InMemoryFeedbackStorage()
        self.lineage_tracker = lineage_tracker
        self.processor = FeedbackProcessor(pipeline)
        
        self.feedback_handlers = {
            "embedding": self._handle_embedding_feedback,
            "retrieval": self._handle_retrieval_feedback,
            "reasoning": self._handle_reasoning_feedback,
            "transformation": self._handle_transformation_feedback,
        }
    
    def add_feedback(
        self,
        feedback_type: str,
        content: Dict[str, Any],
        user_id: Optional[str] = None,
        target_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a feedback item.
        
        Args:
            feedback_type: Type of feedback
            content: Content of the feedback
            user_id: Optional identifier for the user providing feedback
            target_id: Optional identifier for the target of the feedback
            metadata: Additional metadata about the feedback
            
        Returns:
            ID of the newly created feedback item
        """
        feedback_id = str(uuid.uuid4())
        
        feedback_item = FeedbackItem(
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            content=content,
            user_id=user_id,
            target_id=target_id,
            metadata=metadata,
        )
        
        self.feedback_items[feedback_id] = feedback_item
        
        # Store feedback
        self.storage_backend.save_feedback(feedback_id, feedback_item.to_dict())
        
        # Record lineage if available
        if self.lineage_tracker and target_id:
            self._record_feedback_lineage(feedback_item)
        
        # Process feedback if handler is available
        if feedback_type in self.feedback_handlers:
            try:
                self.feedback_handlers[feedback_type](feedback_item)
            except Exception as e:
                logger.warning(f"Error processing {feedback_type} feedback: {e}")
        
        return feedback_id
    
    def add_embedding_feedback(
        self,
        rating: int,
        text: str,
        embedding_id: str,
        suggestions: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add embedding feedback.
        
        Args:
            rating: User rating (1-5)
            text: The text that was embedded
            embedding_id: ID of the embedding
            suggestions: Optional suggestions for improvement
            user_id: Optional identifier for the user providing feedback
            metadata: Additional metadata about the feedback
            
        Returns:
            ID of the newly created feedback item
        """
        feedback = EmbeddingFeedback(
            rating=rating,
            text=text,
            embedding_id=embedding_id,
            suggestions=suggestions,
            user_id=user_id,
            metadata=metadata,
        )
        
        return self.add_feedback(
            feedback_type=feedback.feedback_type,
            content=feedback.content,
            user_id=feedback.user_id,
            target_id=feedback.target_id,
            metadata=feedback.metadata,
        )
    
    def add_retrieval_feedback(
        self,
        query: str,
        relevant_results: List[str],
        irrelevant_results: List[str],
        query_id: str,
        missing_results: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add retrieval feedback.
        
        Args:
            query: The query that was searched
            relevant_results: IDs of results deemed relevant
            irrelevant_results: IDs of results deemed irrelevant
            query_id: ID of the query
            missing_results: Optional IDs of results that should have been included
            user_id: Optional identifier for the user providing feedback
            metadata: Additional metadata about the feedback
            
        Returns:
            ID of the newly created feedback item
        """
        feedback = RetrievalFeedback(
            query=query,
            relevant_results=relevant_results,
            irrelevant_results=irrelevant_results,
            query_id=query_id,
            missing_results=missing_results,
            user_id=user_id,
            metadata=metadata,
        )
        
        return self.add_feedback(
            feedback_type=feedback.feedback_type,
            content=feedback.content,
            user_id=feedback.user_id,
            target_id=feedback.target_id,
            metadata=feedback.metadata,
        )
    
    def add_reasoning_feedback(
        self,
        rating: int,
        reasoning_id: str,
        corrections: Optional[List[Dict[str, Any]]] = None,
        explanation: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add reasoning feedback.
        
        Args:
            rating: User rating (1-5)
            reasoning_id: ID of the reasoning process
            corrections: Optional corrections to reasoning steps
            explanation: Optional explanation of the feedback
            user_id: Optional identifier for the user providing feedback
            metadata: Additional metadata about the feedback
            
        Returns:
            ID of the newly created feedback item
        """
        feedback = ReasoningFeedback(
            rating=rating,
            reasoning_id=reasoning_id,
            corrections=corrections,
            explanation=explanation,
            user_id=user_id,
            metadata=metadata,
        )
        
        return self.add_feedback(
            feedback_type=feedback.feedback_type,
            content=feedback.content,
            user_id=feedback.user_id,
            target_id=feedback.target_id,
            metadata=feedback.metadata,
        )
    
    def add_transformation_feedback(
        self,
        rating: int,
        stage: str,
        transformation_id: str,
        suggestions: Optional[List[str]] = None,
        example_inputs: Optional[List[Any]] = None,
        example_expected_outputs: Optional[List[Any]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add transformation feedback.
        
        Args:
            rating: User rating (1-5)
            stage: The pipeline stage this feedback relates to
            transformation_id: ID of the transformation
            suggestions: Optional suggestions for improvement
            example_inputs: Optional example inputs for the transformation
            example_expected_outputs: Optional expected outputs for the example inputs
            user_id: Optional identifier for the user providing feedback
            metadata: Additional metadata about the feedback
            
        Returns:
            ID of the newly created feedback item
        """
        feedback = TransformationFeedback(
            rating=rating,
            stage=stage,
            transformation_id=transformation_id,
            suggestions=suggestions,
            example_inputs=example_inputs,
            example_expected_outputs=example_expected_outputs,
            user_id=user_id,
            metadata=metadata,
        )
        
        return self.add_feedback(
            feedback_type=feedback.feedback_type,
            content=feedback.content,
            user_id=feedback.user_id,
            target_id=feedback.target_id,
            metadata=feedback.metadata,
        )
    
    def get_feedback(self, feedback_id: str) -> Optional[FeedbackItem]:
        """
        Get a feedback item by ID.
        
        Args:
            feedback_id: The ID of the feedback item to retrieve
            
        Returns:
            The feedback item or None if not found
        """
        if feedback_id in self.feedback_items:
            return self.feedback_items[feedback_id]
        
        # Try to load from storage
        feedback_data = self.storage_backend.load_feedback(feedback_id)
        if feedback_data:
            feedback_item = FeedbackItem.from_dict(feedback_data)
            self.feedback_items[feedback_id] = feedback_item
            return feedback_item
        
        return None
    
    def get_feedback_by_target(
        self, target_id: str, feedback_type: Optional[str] = None
    ) -> List[FeedbackItem]:
        """
        Get feedback items by target ID.
        
        Args:
            target_id: Target ID to filter by
            feedback_type: Optional feedback type to filter by
            
        Returns:
            List of feedback items for the specified target
        """
        result = []
        
        # Load from storage
        feedback_data_list = self.storage_backend.load_feedback_by_target(target_id, feedback_type)
        for feedback_data in feedback_data_list:
            feedback_id = feedback_data["feedback_id"]
            if feedback_id not in self.feedback_items:
                feedback_item = FeedbackItem.from_dict(feedback_data)
                self.feedback_items[feedback_id] = feedback_item
                result.append(feedback_item)
            else:
                result.append(self.feedback_items[feedback_id])
        
        return result
    
    def get_feedback_by_user(
        self, user_id: str, feedback_type: Optional[str] = None
    ) -> List[FeedbackItem]:
        """
        Get feedback items by user ID.
        
        Args:
            user_id: User ID to filter by
            feedback_type: Optional feedback type to filter by
            
        Returns:
            List of feedback items from the specified user
        """
        result = []
        
        # Load from storage
        feedback_data_list = self.storage_backend.load_feedback_by_user(user_id, feedback_type)
        for feedback_data in feedback_data_list:
            feedback_id = feedback_data["feedback_id"]
            if feedback_id not in self.feedback_items:
                feedback_item = FeedbackItem.from_dict(feedback_data)
                self.feedback_items[feedback_id] = feedback_item
                result.append(feedback_item)
            else:
                result.append(self.feedback_items[feedback_id])
        
        return result
    
    def get_feedback_stats(
        self, feedback_type: Optional[str] = None, timeframe: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about collected feedback.
        
        Args:
            feedback_type: Optional feedback type to filter by
            timeframe: Optional (start_time, end_time) tuple to filter by
            
        Returns:
            Statistics about the collected feedback
        """
        # Get the processor to calculate statistics
        if feedback_type == "embedding":
            return self.processor.get_embedding_quality_metrics()
        elif feedback_type == "retrieval":
            return self.processor.get_retrieval_performance_metrics()
        
        # For other types or all types, use the processor to calculate comprehensive stats
        stats = {
            "total_feedback": len(self.feedback_items),
            "by_type": {},
            "by_rating": {},
            "recent_trends": {},
            "improvement_recommendations": self.processor.get_improvement_recommendations(),
        }
        
        # Count by type
        for item in self.feedback_items.values():
            if feedback_type and item.feedback_type != feedback_type:
                continue
            
            if timeframe:
                start_time, end_time = timeframe
                if not (start_time <= item.timestamp <= end_time):
                    continue
            
            # By type
            if item.feedback_type not in stats["by_type"]:
                stats["by_type"][item.feedback_type] = 0
            stats["by_type"][item.feedback_type] += 1
            
            # By rating (if available)
            rating = item.content.get("rating")
            if rating:
                if rating not in stats["by_rating"]:
                    stats["by_rating"][rating] = 0
                stats["by_rating"][rating] += 1
        
        # Calculate time-based trends
        if self.feedback_items:
            items_by_date = {}
            for item in self.feedback_items.values():
                if feedback_type and item.feedback_type != feedback_type:
                    continue
                
                if timeframe:
                    start_time, end_time = timeframe
                    if not (start_time <= item.timestamp <= end_time):
                        continue
                
                date = datetime.date.fromtimestamp(item.timestamp).isoformat()
                if date not in items_by_date:
                    items_by_date[date] = 0
                items_by_date[date] += 1
            
            # Sort by date
            stats["recent_trends"] = {
                date: count for date, count in sorted(items_by_date.items())
            }
        
        return stats
    
    def delete_feedback(self, feedback_id: str) -> bool:
        """
        Delete a feedback item.
        
        Args:
            feedback_id: ID of the feedback item to delete
            
        Returns:
            True if the feedback was deleted successfully, False otherwise
        """
        # Delete from storage
        success = self.storage_backend.delete_feedback(feedback_id)
        
        # Delete from memory if successful
        if success and feedback_id in self.feedback_items:
            del self.feedback_items[feedback_id]
        
        return success
    
    def _record_feedback_lineage(self, feedback_item: FeedbackItem) -> Optional[str]:
        """
        Record feedback lineage.
        
        Args:
            feedback_item: Feedback item to record lineage for
            
        Returns:
            ID of the recorded lineage event, or None if recording failed
        """
        if not self.lineage_tracker:
            return None
        
        try:
            # Create event for feedback
            event_id = self.lineage_tracker.record_event(
                event_type="user_feedback",
                operation_name=f"{feedback_item.feedback_type}_feedback",
                inputs={"target_id": feedback_item.target_id},
                outputs={"feedback_id": feedback_item.feedback_id},
                metadata={
                    "feedback_type": feedback_item.feedback_type,
                    "user_id": feedback_item.user_id,
                    "timestamp": feedback_item.timestamp,
                    "content_summary": {
                        k: v for k, v in feedback_item.content.items()
                        if k not in ["text", "query"]  # Avoid storing full text/query
                    },
                },
                parent_event_ids=[feedback_item.target_id] if feedback_item.target_id else [],
            )
            
            return event_id
        except Exception as e:
            logger.warning(f"Error recording feedback lineage: {e}")
            return None
    
    def _handle_embedding_feedback(self, feedback_item: FeedbackItem) -> None:
        """
        Handle embedding feedback.
        
        Args:
            feedback_item: The feedback item to handle
        """
        try:
            self.processor.process_embedding_feedback(feedback_item)
        except Exception as e:
            logger.warning(f"Error handling embedding feedback: {e}")
    
    def _handle_retrieval_feedback(self, feedback_item: FeedbackItem) -> None:
        """
        Handle retrieval feedback.
        
        Args:
            feedback_item: The feedback item to handle
        """
        try:
            self.processor.process_retrieval_feedback(feedback_item)
        except Exception as e:
            logger.warning(f"Error handling retrieval feedback: {e}")
    
    def _handle_reasoning_feedback(self, feedback_item: FeedbackItem) -> None:
        """
        Handle reasoning feedback.
        
        Args:
            feedback_item: The feedback item to handle
        """
        try:
            self.processor.process_reasoning_feedback(feedback_item)
        except Exception as e:
            logger.warning(f"Error handling reasoning feedback: {e}")
    
    def _handle_transformation_feedback(self, feedback_item: FeedbackItem) -> None:
        """
        Handle transformation feedback.
        
        Args:
            feedback_item: The feedback item to handle
        """
        try:
            self.processor.process_transformation_feedback(feedback_item)
        except Exception as e:
            logger.warning(f"Error handling transformation feedback: {e}")
    
    def register_feedback_handler(
        self, feedback_type: str, handler: Callable[[FeedbackItem], None]
    ) -> None:
        """
        Register a handler for a specific feedback type.
        
        Args:
            feedback_type: The type of feedback to handle
            handler: The handler function
        """
        self.feedback_handlers[feedback_type] = handler
    
    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations for system improvements based on feedback.
        
        Returns:
            List of improvement recommendations
        """
        return self.processor.get_improvement_recommendations()


class FeedbackVisualization:
    """
    Visualization tools for feedback analysis.
    
    Provides methods for generating visualizations of feedback data.
    """
    
    @staticmethod
    def generate_feedback_summary(feedback_collector: UserFeedbackCollector) -> Dict[str, Any]:
        """
        Generate a summary of feedback for visualization.
        
        Args:
            feedback_collector: The feedback collector to generate summary for
            
        Returns:
            Dictionary of summary data suitable for visualization
        """
        # Get overall stats
        stats = feedback_collector.get_feedback_stats()
        
        # Get type-specific stats
        embedding_stats = feedback_collector.get_feedback_stats(feedback_type="embedding")
        retrieval_stats = feedback_collector.get_feedback_stats(feedback_type="retrieval")
        
        # Get recommendations
        recommendations = feedback_collector.get_improvement_recommendations()
        
        # Build summary
        summary = {
            "overview": {
                "total_feedback": stats.get("total_feedback", 0),
                "by_type": stats.get("by_type", {}),
                "recent_trends": stats.get("recent_trends", {}),
            },
            "embedding_quality": {
                "avg_rating": embedding_stats.get("avg_rating", 0),
                "rating_distribution": embedding_stats.get("rating_distribution", {}),
                "embeddings": embedding_stats.get("embeddings", 0),
            },
            "retrieval_performance": {
                "avg_precision": retrieval_stats.get("avg_precision", 0),
                "avg_recall": retrieval_stats.get("avg_recall", 0),
                "avg_f1": retrieval_stats.get("avg_f1", 0),
                "queries": retrieval_stats.get("queries", 0),
            },
            "recommendations": [
                {
                    "type": rec["type"],
                    "priority": rec["priority"],
                    "description": rec["description"],
                }
                for rec in recommendations
            ],
        }
        
        return summary
    
    @staticmethod
    def format_feedback_for_display(feedback_item: FeedbackItem) -> Dict[str, Any]:
        """
        Format a feedback item for display.
        
        Args:
            feedback_item: The feedback item to format
            
        Returns:
            Dictionary of formatted feedback data
        """
        # Format basic information
        formatted = {
            "id": feedback_item.feedback_id,
            "type": feedback_item.feedback_type,
            "date": datetime.datetime.fromtimestamp(feedback_item.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "user": feedback_item.user_id or "Anonymous",
            "content": {},
        }
        
        # Format content based on feedback type
        if feedback_item.feedback_type == "embedding":
            rating = feedback_item.content.get("rating", 0)
            formatted["content"] = {
                "rating": rating,
                "rating_text": ["Very Poor", "Poor", "Average", "Good", "Excellent"][rating - 1] if 1 <= rating <= 5 else "Unknown",
                "text": feedback_item.content.get("text", "")[:100] + "...",
                "suggestions": feedback_item.content.get("suggestions", []),
            }
        
        elif feedback_item.feedback_type == "retrieval":
            formatted["content"] = {
                "query": feedback_item.content.get("query", "")[:100] + "...",
                "relevant_count": len(feedback_item.content.get("relevant_results", [])),
                "irrelevant_count": len(feedback_item.content.get("irrelevant_results", [])),
                "missing_count": len(feedback_item.content.get("missing_results", [])),
            }
        
        elif feedback_item.feedback_type == "reasoning":
            rating = feedback_item.content.get("rating", 0)
            formatted["content"] = {
                "rating": rating,
                "rating_text": ["Very Poor", "Poor", "Average", "Good", "Excellent"][rating - 1] if 1 <= rating <= 5 else "Unknown",
                "correction_count": len(feedback_item.content.get("corrections", [])),
                "explanation": feedback_item.content.get("explanation", "")[:100] + "..." if feedback_item.content.get("explanation", "") else "",
            }
        
        elif feedback_item.feedback_type == "transformation":
            rating = feedback_item.content.get("rating", 0)
            formatted["content"] = {
                "rating": rating,
                "rating_text": ["Very Poor", "Poor", "Average", "Good", "Excellent"][rating - 1] if 1 <= rating <= 5 else "Unknown",
                "stage": feedback_item.content.get("stage", ""),
                "suggestion_count": len(feedback_item.content.get("suggestions", [])),
                "example_count": len(feedback_item.content.get("example_inputs", [])),
            }
        
        return formatted