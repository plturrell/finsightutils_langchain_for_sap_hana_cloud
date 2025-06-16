"""
Information fingerprinting module for tracking data lineage in transformations.

This module provides tools for creating, tracking, and comparing fingerprints
of information as it transforms through the vector pipeline, allowing for
traceability and origin verification.
"""

import hashlib
import uuid
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)


class InformationSignature:
    """
    Represents a digital signature of a piece of information.
    
    Captures essential characteristics of information that can be
    used to identify it throughout transformations.
    """
    
    def __init__(
        self,
        signature_id: str,
        content_hash: str,
        feature_vector: List[float],
        metadata: Dict[str, Any],
        source_id: str,
        timestamp: Optional[float] = None,
    ):
        """
        Initialize an information signature.
        
        Args:
            signature_id: Unique identifier for this signature
            content_hash: Hash of the content
            feature_vector: Feature vector representation
            metadata: Additional metadata about the signature
            source_id: Identifier for the source of the information
            timestamp: Time when this signature was created
        """
        self.signature_id = signature_id
        self.content_hash = content_hash
        self.feature_vector = feature_vector
        self.metadata = metadata
        self.source_id = source_id
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the information signature to a dictionary."""
        return {
            "signature_id": self.signature_id,
            "content_hash": self.content_hash,
            "feature_vector": self.feature_vector,
            "metadata": self.metadata,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InformationSignature":
        """Create an information signature from a dictionary."""
        return cls(
            signature_id=data["signature_id"],
            content_hash=data["content_hash"],
            feature_vector=data["feature_vector"],
            metadata=data["metadata"],
            source_id=data["source_id"],
            timestamp=data.get("timestamp", time.time()),
        )
    
    def similarity(self, other: "InformationSignature") -> float:
        """
        Calculate similarity with another signature.
        
        Args:
            other: The other signature to compare with
            
        Returns:
            Similarity score (0-1)
        """
        # Exact hash match
        if self.content_hash == other.content_hash:
            return 1.0
        
        # Feature vector similarity
        if self.feature_vector and other.feature_vector:
            return self._cosine_similarity(self.feature_vector, other.feature_vector)
        
        # Default similarity based on metadata
        metadata_sim = self._metadata_similarity(self.metadata, other.metadata)
        return metadata_sim
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        if len(a) != len(b):
            # If vectors have different dimensions, truncate to the shorter length
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]
        
        a_np = np.array(a)
        b_np = np.array(b)
        
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        cos_sim = np.dot(a_np, b_np) / (norm_a * norm_b)
        return max(0.0, min(1.0, cos_sim))  # Clamp to [0, 1]
    
    def _metadata_similarity(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """
        Calculate similarity between metadata dictionaries.
        
        Args:
            a: First metadata dictionary
            b: Second metadata dictionary
            
        Returns:
            Similarity score (0-1)
        """
        if not a or not b:
            return 0.0
        
        # Get common keys
        common_keys = set(a.keys()).intersection(set(b.keys()))
        if not common_keys:
            return 0.0
        
        # Calculate similarity for each common key
        similarities = []
        for key in common_keys:
            if a[key] == b[key]:
                similarities.append(1.0)
            elif isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
                # Normalize numerical values to [0, 1]
                max_val = max(abs(a[key]), abs(b[key]))
                if max_val == 0:
                    similarities.append(1.0)
                else:
                    similarities.append(1.0 - abs(a[key] - b[key]) / max_val)
            elif isinstance(a[key], str) and isinstance(b[key], str):
                # Jaccard similarity for strings
                words_a = set(a[key].lower().split())
                words_b = set(b[key].lower().split())
                if not words_a or not words_b:
                    similarities.append(0.0)
                else:
                    intersection = words_a.intersection(words_b)
                    union = words_a.union(words_b)
                    similarities.append(len(intersection) / len(union))
            else:
                # Default for incomparable types
                similarities.append(0.0)
        
        # Average similarity across all common keys
        return sum(similarities) / len(similarities)


class InformationFingerprint:
    """
    Represents a composite fingerprint of a piece of information.
    
    Combines multiple signatures to represent complex information
    and its transformation history.
    """
    
    def __init__(
        self,
        fingerprint_id: str,
        signatures: List[InformationSignature],
        transformation_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Initialize an information fingerprint.
        
        Args:
            fingerprint_id: Unique identifier for this fingerprint
            signatures: List of information signatures
            transformation_history: List of transformation events
            metadata: Additional metadata about the fingerprint
            timestamp: Time when this fingerprint was created
        """
        self.fingerprint_id = fingerprint_id
        self.signatures = signatures
        self.transformation_history = transformation_history or []
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the information fingerprint to a dictionary."""
        return {
            "fingerprint_id": self.fingerprint_id,
            "signatures": [sig.to_dict() for sig in self.signatures],
            "transformation_history": self.transformation_history,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InformationFingerprint":
        """Create an information fingerprint from a dictionary."""
        signatures = [
            InformationSignature.from_dict(sig_data)
            for sig_data in data["signatures"]
        ]
        
        return cls(
            fingerprint_id=data["fingerprint_id"],
            signatures=signatures,
            transformation_history=data.get("transformation_history", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )
    
    def add_transformation(
        self,
        transformation_type: str,
        transformation_parameters: Dict[str, Any],
        input_fingerprint_id: Optional[str] = None,
    ) -> None:
        """
        Add a transformation event to the history.
        
        Args:
            transformation_type: Type of transformation
            transformation_parameters: Parameters of the transformation
            input_fingerprint_id: ID of the input fingerprint
        """
        self.transformation_history.append({
            "transformation_type": transformation_type,
            "transformation_parameters": transformation_parameters,
            "input_fingerprint_id": input_fingerprint_id,
            "timestamp": time.time(),
        })
    
    def similarity(self, other: "InformationFingerprint") -> float:
        """
        Calculate similarity with another fingerprint.
        
        Args:
            other: The other fingerprint to compare with
            
        Returns:
            Similarity score (0-1)
        """
        if not self.signatures or not other.signatures:
            return 0.0
        
        # Calculate all pairwise similarities
        similarities = []
        for sig1 in self.signatures:
            for sig2 in other.signatures:
                similarities.append(sig1.similarity(sig2))
        
        # Return average of top N similarities
        top_n = min(len(self.signatures), len(other.signatures))
        similarities.sort(reverse=True)
        return sum(similarities[:top_n]) / top_n
    
    def sources(self) -> Set[str]:
        """
        Get the set of source IDs represented in this fingerprint.
        
        Returns:
            Set of source IDs
        """
        return {sig.source_id for sig in self.signatures}
    
    def merge(self, other: "InformationFingerprint") -> "InformationFingerprint":
        """
        Merge with another fingerprint.
        
        Args:
            other: The other fingerprint to merge with
            
        Returns:
            A new merged fingerprint
        """
        new_fingerprint_id = str(uuid.uuid4())
        
        # Combine signatures (avoid duplicates by content hash)
        content_hashes = {sig.content_hash for sig in self.signatures}
        combined_signatures = list(self.signatures)
        
        for sig in other.signatures:
            if sig.content_hash not in content_hashes:
                combined_signatures.append(sig)
                content_hashes.add(sig.content_hash)
        
        # Combine transformation history
        combined_history = list(self.transformation_history)
        for event in other.transformation_history:
            if event not in combined_history:
                combined_history.append(event)
        
        # Combine metadata
        combined_metadata = {**self.metadata, **other.metadata}
        
        # Add merge event to transformation history
        merge_event = {
            "transformation_type": "merge",
            "transformation_parameters": {
                "merged_fingerprint_ids": [self.fingerprint_id, other.fingerprint_id],
            },
            "timestamp": time.time(),
        }
        combined_history.append(merge_event)
        
        return InformationFingerprint(
            fingerprint_id=new_fingerprint_id,
            signatures=combined_signatures,
            transformation_history=combined_history,
            metadata=combined_metadata,
        )


class FingerprintManager:
    """
    Manages information fingerprints.
    
    Provides tools for creating, tracking, and comparing fingerprints
    throughout the data transformation pipeline.
    """
    
    def __init__(self, storage_backend=None):
        """
        Initialize a fingerprint manager.
        
        Args:
            storage_backend: Optional backend for storing fingerprints
        """
        self.fingerprints = {}
        self.storage_backend = storage_backend
    
    def create_signature(
        self,
        content: Any,
        source_id: str,
        feature_vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InformationSignature:
        """
        Create a signature for a piece of information.
        
        Args:
            content: The content to create a signature for
            source_id: Identifier for the source of the information
            feature_vector: Optional feature vector representation
            metadata: Additional metadata about the signature
            
        Returns:
            The created signature
        """
        signature_id = str(uuid.uuid4())
        
        # Generate content hash
        content_hash = self._hash_content(content)
        
        # Generate feature vector if not provided
        if feature_vector is None:
            feature_vector = self._generate_feature_vector(content)
        
        # Create signature
        signature = InformationSignature(
            signature_id=signature_id,
            content_hash=content_hash,
            feature_vector=feature_vector,
            metadata=metadata or {},
            source_id=source_id,
        )
        
        return signature
    
    def create_fingerprint(
        self,
        content: Any,
        source_id: str,
        chunk_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InformationFingerprint:
        """
        Create a fingerprint for a piece of information.
        
        Args:
            content: The content to create a fingerprint for
            source_id: Identifier for the source of the information
            chunk_size: Optional size for chunking content
            metadata: Additional metadata about the fingerprint
            
        Returns:
            The created fingerprint
        """
        fingerprint_id = str(uuid.uuid4())
        
        # Create signatures
        signatures = []
        
        if chunk_size and isinstance(content, str) and len(content) > chunk_size:
            # Chunk content and create a signature for each chunk
            chunks = self._chunk_text(content, chunk_size)
            for i, chunk in enumerate(chunks):
                chunk_source_id = f"{source_id}_chunk_{i}"
                chunk_metadata = {
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    **metadata or {},
                }
                signature = self.create_signature(
                    content=chunk,
                    source_id=chunk_source_id,
                    metadata=chunk_metadata,
                )
                signatures.append(signature)
        else:
            # Create a single signature
            signature = self.create_signature(
                content=content,
                source_id=source_id,
                metadata=metadata or {},
            )
            signatures.append(signature)
        
        # Create fingerprint
        fingerprint = InformationFingerprint(
            fingerprint_id=fingerprint_id,
            signatures=signatures,
            metadata=metadata or {},
        )
        
        # Store fingerprint
        self.fingerprints[fingerprint_id] = fingerprint
        
        if self.storage_backend:
            self.storage_backend.save_fingerprint(fingerprint_id, fingerprint.to_dict())
        
        return fingerprint
    
    def transform_fingerprint(
        self,
        fingerprint_id: str,
        transformation_type: str,
        transformation_parameters: Dict[str, Any],
        output_content: Any,
        output_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Transform a fingerprint and create a new one.
        
        Args:
            fingerprint_id: ID of the fingerprint to transform
            transformation_type: Type of transformation
            transformation_parameters: Parameters of the transformation
            output_content: Output content after transformation
            output_metadata: Metadata for the output fingerprint
            
        Returns:
            ID of the new fingerprint
            
        Raises:
            ValueError: If the fingerprint_id is not found
        """
        # Get the input fingerprint
        input_fingerprint = self.get_fingerprint(fingerprint_id)
        if not input_fingerprint:
            raise ValueError(f"Fingerprint {fingerprint_id} not found")
        
        # Create a new fingerprint for the output
        new_fingerprint_id = str(uuid.uuid4())
        
        # Create signatures for the output content
        output_signatures = []
        
        if isinstance(output_content, str) and transformation_parameters.get("chunk_size"):
            # Chunk output content and create a signature for each chunk
            chunk_size = transformation_parameters["chunk_size"]
            chunks = self._chunk_text(output_content, chunk_size)
            
            for i, chunk in enumerate(chunks):
                chunk_source_id = f"{fingerprint_id}_transformed_{i}"
                chunk_metadata = {
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "transformation_type": transformation_type,
                    **output_metadata or {},
                }
                signature = self.create_signature(
                    content=chunk,
                    source_id=chunk_source_id,
                    metadata=chunk_metadata,
                )
                output_signatures.append(signature)
        else:
            # Create a single signature for the output content
            output_signature = self.create_signature(
                content=output_content,
                source_id=fingerprint_id,  # Original fingerprint as source
                metadata={
                    "transformation_type": transformation_type,
                    **output_metadata or {},
                },
            )
            output_signatures.append(output_signature)
        
        # Create the output fingerprint
        output_fingerprint = InformationFingerprint(
            fingerprint_id=new_fingerprint_id,
            signatures=output_signatures,
            metadata=output_metadata or {},
        )
        
        # Add the transformation to the history
        output_fingerprint.add_transformation(
            transformation_type=transformation_type,
            transformation_parameters=transformation_parameters,
            input_fingerprint_id=fingerprint_id,
        )
        
        # Store the output fingerprint
        self.fingerprints[new_fingerprint_id] = output_fingerprint
        
        if self.storage_backend:
            self.storage_backend.save_fingerprint(new_fingerprint_id, output_fingerprint.to_dict())
        
        return new_fingerprint_id
    
    def get_fingerprint(self, fingerprint_id: str) -> Optional[InformationFingerprint]:
        """
        Get a fingerprint by ID.
        
        Args:
            fingerprint_id: The ID of the fingerprint to retrieve
            
        Returns:
            The fingerprint or None if not found
        """
        if fingerprint_id in self.fingerprints:
            return self.fingerprints[fingerprint_id]
        
        # Try to load from storage backend
        if self.storage_backend:
            fingerprint_data = self.storage_backend.load_fingerprint(fingerprint_id)
            if fingerprint_data:
                fingerprint = InformationFingerprint.from_dict(fingerprint_data)
                self.fingerprints[fingerprint_id] = fingerprint
                return fingerprint
        
        return None
    
    def find_similar_fingerprints(
        self,
        fingerprint_id: str,
        threshold: float = 0.7,
        max_results: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find fingerprints similar to the given one.
        
        Args:
            fingerprint_id: ID of the fingerprint to compare with
            threshold: Minimum similarity threshold (0-1)
            max_results: Maximum number of results to return
            
        Returns:
            List of (fingerprint_id, similarity) tuples
            
        Raises:
            ValueError: If the fingerprint_id is not found
        """
        # Get the reference fingerprint
        reference = self.get_fingerprint(fingerprint_id)
        if not reference:
            raise ValueError(f"Fingerprint {fingerprint_id} not found")
        
        # Calculate similarities with all other fingerprints
        similarities = []
        for fid, fingerprint in self.fingerprints.items():
            if fid != fingerprint_id:
                similarity = reference.similarity(fingerprint)
                if similarity >= threshold:
                    similarities.append((fid, similarity))
        
        # Sort by similarity (descending) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def trace_lineage(self, fingerprint_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Trace the lineage of a fingerprint.
        
        Args:
            fingerprint_id: ID of the fingerprint to trace
            max_depth: Maximum depth to trace
            
        Returns:
            Lineage information
            
        Raises:
            ValueError: If the fingerprint_id is not found
        """
        # Get the fingerprint
        fingerprint = self.get_fingerprint(fingerprint_id)
        if not fingerprint:
            raise ValueError(f"Fingerprint {fingerprint_id} not found")
        
        # Build lineage tree
        lineage = {
            "fingerprint_id": fingerprint_id,
            "metadata": fingerprint.metadata,
            "sources": list(fingerprint.sources()),
            "transformation_history": fingerprint.transformation_history,
            "ancestors": [],
        }
        
        # Trace ancestors
        current_depth = 0
        ancestor_ids = self._get_ancestor_ids(fingerprint)
        
        while ancestor_ids and current_depth < max_depth:
            ancestors = []
            next_ancestor_ids = []
            
            for aid in ancestor_ids:
                ancestor = self.get_fingerprint(aid)
                if ancestor:
                    ancestor_info = {
                        "fingerprint_id": aid,
                        "metadata": ancestor.metadata,
                        "sources": list(ancestor.sources()),
                        "transformation_history": ancestor.transformation_history,
                        "ancestors": [],
                    }
                    ancestors.append(ancestor_info)
                    
                    # Get next level of ancestors
                    next_ancestor_ids.extend(self._get_ancestor_ids(ancestor))
            
            lineage["ancestors"] = ancestors
            ancestor_ids = next_ancestor_ids
            current_depth += 1
        
        return lineage
    
    def _get_ancestor_ids(self, fingerprint: InformationFingerprint) -> List[str]:
        """
        Get the IDs of a fingerprint's ancestors.
        
        Args:
            fingerprint: The fingerprint to get ancestors for
            
        Returns:
            List of ancestor fingerprint IDs
        """
        ancestor_ids = []
        
        for event in fingerprint.transformation_history:
            if event.get("input_fingerprint_id"):
                ancestor_ids.append(event["input_fingerprint_id"])
            elif event.get("transformation_type") == "merge" and "merged_fingerprint_ids" in event.get("transformation_parameters", {}):
                ancestor_ids.extend(event["transformation_parameters"]["merged_fingerprint_ids"])
        
        return ancestor_ids
    
    def _hash_content(self, content: Any) -> str:
        """
        Generate a hash for the content.
        
        Args:
            content: The content to hash
            
        Returns:
            Content hash
        """
        if content is None:
            return hashlib.sha256(b"none").hexdigest()
        
        if isinstance(content, str):
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        if isinstance(content, bytes):
            return hashlib.sha256(content).hexdigest()
        
        if isinstance(content, (list, tuple)):
            # Hash each item and combine
            item_hashes = [self._hash_content(item) for item in content]
            return hashlib.sha256(''.join(item_hashes).encode('utf-8')).hexdigest()
        
        if isinstance(content, dict):
            # Sort keys and hash each key-value pair
            sorted_items = sorted(content.items())
            item_hashes = [
                self._hash_content(key) + self._hash_content(value)
                for key, value in sorted_items
            ]
            return hashlib.sha256(''.join(item_hashes).encode('utf-8')).hexdigest()
        
        # For other types, convert to string first
        return hashlib.sha256(str(content).encode('utf-8')).hexdigest()
    
    def _generate_feature_vector(self, content: Any) -> List[float]:
        """
        Generate a feature vector for the content.
        
        Args:
            content: The content to generate a feature vector for
            
        Returns:
            Feature vector
        """
        # This is a simplified implementation
        # A full implementation would use more sophisticated NLP
        
        if isinstance(content, str):
            # Simple character frequency vector
            char_freqs = [0] * 128  # ASCII characters
            
            if content:
                for char in content:
                    code = ord(char)
                    if 0 <= code < 128:
                        char_freqs[code] += 1
                
                # Normalize
                total = sum(char_freqs)
                if total > 0:
                    char_freqs = [freq / total for freq in char_freqs]
            
            return char_freqs
        
        # For non-string content, return a simple hash-based vector
        content_hash = self._hash_content(content)
        hash_bytes = bytes.fromhex(content_hash)
        
        # Convert hash bytes to float vector
        feature_vector = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                # Convert 4 bytes to a float between 0 and 1
                value = int.from_bytes(chunk, byteorder='big') / (2**32 - 1)
                feature_vector.append(value)
        
        return feature_vector
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split by sentences and then combine into chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Start a new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks