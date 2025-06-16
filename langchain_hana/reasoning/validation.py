"""
Reasoning validation module for validating reasoning based on SimpleQA and GSM-Symbolic.

This module provides tools for validating reasoning paths and calculations
in language model responses, inspired by SimpleQA and GSM-Symbolic methodologies.
"""

import re
import uuid
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import numpy as np

logger = logging.getLogger(__name__)


class ValidationResult:
    """
    Represents the result of a reasoning validation.
    
    Captures the validation outcome, details, and confidence.
    """
    
    def __init__(
        self,
        result_id: str,
        is_valid: bool,
        confidence: float,
        details: Dict[str, Any],
        validation_type: str,
        timestamp: Optional[float] = None,
    ):
        """
        Initialize a validation result.
        
        Args:
            result_id: Unique identifier for this result
            is_valid: Whether the reasoning is valid
            confidence: Confidence in the validation result (0-1)
            details: Detailed validation information
            validation_type: Type of validation performed
            timestamp: Time when this validation was performed
        """
        self.result_id = result_id
        self.is_valid = is_valid
        self.confidence = confidence
        self.details = details
        self.validation_type = validation_type
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "result_id": self.result_id,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "details": self.details,
            "validation_type": self.validation_type,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Create a validation result from a dictionary."""
        return cls(
            result_id=data["result_id"],
            is_valid=data["is_valid"],
            confidence=data["confidence"],
            details=data["details"],
            validation_type=data["validation_type"],
            timestamp=data.get("timestamp", time.time()),
        )


class ReasoningValidator:
    """
    Validates reasoning paths and calculations.
    
    Provides tools for validating reasoning based on SimpleQA and GSM-Symbolic
    methodologies, detecting errors and inconsistencies in reasoning.
    """
    
    def __init__(self):
        """Initialize a reasoning validator."""
        self.validators = {
            "logical_consistency": self._validate_logical_consistency,
            "calculation": self._validate_calculation,
            "citation": self._validate_citation,
            "hallucination": self._validate_hallucination,
            "symbolic_math": self._validate_symbolic_math,
        }
    
    def validate(
        self,
        reasoning_text: str,
        validation_types: Optional[List[str]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ValidationResult]:
        """
        Validate reasoning based on specified validation types.
        
        Args:
            reasoning_text: The reasoning text to validate
            validation_types: Types of validation to perform (default: all)
            ground_truth: Optional ground truth information for validation
            sources: Optional source documents for citation validation
            metadata: Additional metadata for validation
            
        Returns:
            Dictionary of validation results by validation type
        """
        if validation_types is None:
            validation_types = list(self.validators.keys())
        
        results = {}
        
        for validation_type in validation_types:
            if validation_type in self.validators:
                validator = self.validators[validation_type]
                try:
                    result = validator(
                        reasoning_text=reasoning_text,
                        ground_truth=ground_truth,
                        sources=sources,
                        metadata=metadata,
                    )
                    results[validation_type] = result
                except Exception as e:
                    logger.warning(f"Error in {validation_type} validation: {e}")
                    # Create an error result
                    results[validation_type] = ValidationResult(
                        result_id=str(uuid.uuid4()),
                        is_valid=False,
                        confidence=0.0,
                        details={"error": str(e)},
                        validation_type=validation_type,
                    )
            else:
                logger.warning(f"Unknown validation type: {validation_type}")
        
        return results
    
    def _validate_logical_consistency(
        self,
        reasoning_text: str,
        ground_truth: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate the logical consistency of reasoning.
        
        Args:
            reasoning_text: The reasoning text to validate
            ground_truth: Optional ground truth information
            sources: Optional source documents
            metadata: Additional metadata
            
        Returns:
            Validation result
        """
        result_id = str(uuid.uuid4())
        
        # Define patterns for logical inconsistencies
        contradiction_patterns = [
            r"(.*?)\s+but\s+also\s+(.*?)",
            r"(.*?)\s+however\s+(.*?)",
            r"(.*?)\s+is\s+both\s+(.*?)\s+and\s+(.*?)",
            r"(.*?)\s+is\s+not\s+(.*?)\s+but\s+is\s+(.*?)",
        ]
        
        # Check for contradictions
        contradictions = []
        for pattern in contradiction_patterns:
            matches = re.finditer(pattern, reasoning_text, re.IGNORECASE)
            for match in matches:
                # Further analyze the match to determine if it's a true contradiction
                # This is a simplified implementation
                contradictions.append({
                    "pattern": pattern,
                    "match": match.group(0),
                })
        
        # Extract claims and check for logical relationships
        sentences = re.split(r'[.!?]', reasoning_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Check for inconsistent claims
        claims = {}
        inconsistencies = []
        
        # This is a simplified implementation
        # A full implementation would use NLP to extract and compare claims
        
        # Determine validation result
        is_valid = len(contradictions) == 0 and len(inconsistencies) == 0
        confidence = 0.8 if is_valid else 0.6  # Simplified confidence calculation
        
        details = {
            "contradictions": contradictions,
            "inconsistencies": inconsistencies,
            "sentence_count": len(sentences),
        }
        
        return ValidationResult(
            result_id=result_id,
            is_valid=is_valid,
            confidence=confidence,
            details=details,
            validation_type="logical_consistency",
        )
    
    def _validate_calculation(
        self,
        reasoning_text: str,
        ground_truth: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate calculations in reasoning.
        
        Args:
            reasoning_text: The reasoning text to validate
            ground_truth: Optional ground truth information
            sources: Optional source documents
            metadata: Additional metadata
            
        Returns:
            Validation result
        """
        result_id = str(uuid.uuid4())
        
        # Extract calculations
        # This is a simplified implementation
        # A full implementation would use a more sophisticated approach
        calculation_patterns = [
            r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)",
            r"(\d+)\s*-\s*(\d+)\s*=\s*(\d+)",
            r"(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)",
            r"(\d+)\s*/\s*(\d+)\s*=\s*(\d+)",
        ]
        
        calculations = []
        for pattern in calculation_patterns:
            matches = re.finditer(pattern, reasoning_text)
            for match in matches:
                if pattern.count('(') == 3:  # Simple binary operation
                    a = int(match.group(1))
                    b = int(match.group(2))
                    result = int(match.group(3))
                    
                    # Determine the operation
                    if '+' in match.group(0):
                        expected = a + b
                        op = '+'
                    elif '-' in match.group(0):
                        expected = a - b
                        op = '-'
                    elif '*' in match.group(0):
                        expected = a * b
                        op = '*'
                    elif '/' in match.group(0):
                        expected = a / b if b != 0 else float('inf')
                        op = '/'
                    else:
                        continue
                    
                    calculations.append({
                        "expression": match.group(0),
                        "a": a,
                        "b": b,
                        "operation": op,
                        "result": result,
                        "expected": expected,
                        "is_correct": abs(result - expected) < 1e-10,
                    })
        
        # Determine validation result
        errors = [calc for calc in calculations if not calc["is_correct"]]
        is_valid = len(errors) == 0
        
        # Confidence based on the number of calculations and errors
        if len(calculations) == 0:
            confidence = 0.5  # No calculations to validate
        else:
            confidence = 1.0 - len(errors) / len(calculations)
        
        details = {
            "calculations": calculations,
            "errors": errors,
        }
        
        return ValidationResult(
            result_id=result_id,
            is_valid=is_valid,
            confidence=confidence,
            details=details,
            validation_type="calculation",
        )
    
    def _validate_citation(
        self,
        reasoning_text: str,
        ground_truth: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate citations in reasoning.
        
        Args:
            reasoning_text: The reasoning text to validate
            ground_truth: Optional ground truth information
            sources: Optional source documents
            metadata: Additional metadata
            
        Returns:
            Validation result
        """
        result_id = str(uuid.uuid4())
        
        if not sources:
            return ValidationResult(
                result_id=result_id,
                is_valid=False,
                confidence=0.5,
                details={"error": "No sources provided for citation validation"},
                validation_type="citation",
            )
        
        # Extract claims and citations
        # This is a simplified implementation
        citation_patterns = [
            r"(?:according to|as stated in|as mentioned in)\s+(?:the\s+)?(.+)",
            r"(.+?)\s+(?:states|says|mentions|notes|indicates)\s+that\s+(.+)",
            r"\[(\d+)\]",
            r"\((\d+)\)",
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.finditer(pattern, reasoning_text, re.IGNORECASE)
            for match in matches:
                citations.append({
                    "pattern": pattern,
                    "match": match.group(0),
                    "source_reference": match.group(1),
                })
        
        # Extract claims
        sentences = re.split(r'[.!?]', reasoning_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        claims = []
        for sentence in sentences:
            # Check if sentence contains factual claim
            # This is a simplified implementation
            if re.search(r"(?:is|are|was|were|has|have|had|will|would|can|could|should|must)\s+", sentence):
                claims.append({
                    "text": sentence,
                    "has_citation": any(citation["match"] in sentence for citation in citations),
                })
        
        # Check if citations are supported by sources
        valid_citations = []
        invalid_citations = []
        
        for citation in citations:
            source_ref = citation["source_reference"]
            is_valid = False
            
            # Check if citation references a valid source
            # This is a simplified implementation
            for source in sources:
                if (isinstance(source_ref, str) and source_ref.lower() in source.get("title", "").lower()) or \
                   (source_ref.isdigit() and int(source_ref) <= len(sources)):
                    is_valid = True
                    break
            
            if is_valid:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)
        
        # Determine validation result
        uncited_claims = [claim for claim in claims if not claim["has_citation"]]
        is_valid = len(invalid_citations) == 0 and len(uncited_claims) == 0
        
        # Confidence based on the number of claims and citations
        if len(claims) == 0:
            confidence = 0.5  # No claims to validate
        else:
            citation_ratio = len(valid_citations) / max(1, len(claims))
            error_penalty = len(invalid_citations) / max(1, len(citations)) if citations else 0
            confidence = min(1.0, max(0.0, citation_ratio - error_penalty))
        
        details = {
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "uncited_claims": uncited_claims,
            "total_claims": len(claims),
            "total_citations": len(citations),
        }
        
        return ValidationResult(
            result_id=result_id,
            is_valid=is_valid,
            confidence=confidence,
            details=details,
            validation_type="citation",
        )
    
    def _validate_hallucination(
        self,
        reasoning_text: str,
        ground_truth: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate for hallucinations in reasoning.
        
        Args:
            reasoning_text: The reasoning text to validate
            ground_truth: Optional ground truth information
            sources: Optional source documents
            metadata: Additional metadata
            
        Returns:
            Validation result
        """
        result_id = str(uuid.uuid4())
        
        if not sources and not ground_truth:
            return ValidationResult(
                result_id=result_id,
                is_valid=False,
                confidence=0.3,
                details={"error": "No sources or ground truth provided for hallucination validation"},
                validation_type="hallucination",
            )
        
        # Extract factual claims
        # This is a simplified implementation
        claims = []
        sentences = re.split(r'[.!?]', reasoning_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            # Check if sentence contains factual claim
            # This is a simplified implementation
            if re.search(r"(?:is|are|was|were|has|have|had|will|would|can|could|should|must)\s+", sentence) and \
               not re.search(r"(?:I think|in my opinion|perhaps|maybe|might|could be|possibly)", sentence, re.IGNORECASE):
                claims.append({
                    "text": sentence,
                })
        
        # Check claims against sources and ground truth
        supported_claims = []
        unsupported_claims = []
        
        for claim in claims:
            claim_text = claim["text"].lower()
            is_supported = False
            
            # Check claim against sources
            if sources:
                for source in sources:
                    source_text = source.get("text", "").lower()
                    if self._text_similarity(claim_text, source_text) > 0.7:  # Simplified check
                        is_supported = True
                        break
            
            # Check claim against ground truth
            if ground_truth and not is_supported:
                for key, value in ground_truth.items():
                    if isinstance(value, str):
                        if self._text_similarity(claim_text, value.lower()) > 0.7:
                            is_supported = True
                            break
            
            if is_supported:
                supported_claims.append(claim)
            else:
                unsupported_claims.append(claim)
        
        # Determine validation result
        is_valid = len(unsupported_claims) == 0
        
        # Confidence based on the number of claims and support
        if len(claims) == 0:
            confidence = 0.5  # No claims to validate
        else:
            confidence = len(supported_claims) / len(claims)
        
        details = {
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "total_claims": len(claims),
        }
        
        return ValidationResult(
            result_id=result_id,
            is_valid=is_valid,
            confidence=confidence,
            details=details,
            validation_type="hallucination",
        )
    
    def _validate_symbolic_math(
        self,
        reasoning_text: str,
        ground_truth: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate symbolic math in reasoning based on GSM-Symbolic approach.
        
        Args:
            reasoning_text: The reasoning text to validate
            ground_truth: Optional ground truth information
            sources: Optional source documents
            metadata: Additional metadata
            
        Returns:
            Validation result
        """
        result_id = str(uuid.uuid4())
        
        # Extract mathematical expressions
        # This is a simplified implementation
        # A full implementation would use a symbolic math parser
        expressions = []
        math_patterns = [
            r"(\w+)\s*=\s*(.+)",
            r"(\w+)\s*(\+|\-|\*|/|==|!=|>|<|>=|<=)\s*(\w+)",
            r"(\d+(?:\.\d+)?)\s*(\+|\-|\*|/)\s*(\d+(?:\.\d+)?)",
        ]
        
        for pattern in math_patterns:
            matches = re.finditer(pattern, reasoning_text)
            for match in matches:
                expressions.append({
                    "text": match.group(0),
                    "components": match.groups(),
                })
        
        # Check expressions for validity
        # This is a simplified implementation
        valid_expressions = []
        invalid_expressions = []
        variables = {}
        
        for expr in expressions:
            text = expr["text"]
            components = expr["components"]
            
            # Process assignment expressions
            if "=" in text and not "==" in text and not "!=" in text:
                var_name = components[0].strip()
                value_expr = components[1].strip()
                
                try:
                    # Evaluate the right-hand side using existing variables
                    # This is a simplified implementation
                    value = eval(value_expr, {"__builtins__": {}}, variables)
                    variables[var_name] = value
                    valid_expressions.append({
                        "text": text,
                        "type": "assignment",
                        "variable": var_name,
                        "value": value,
                    })
                except Exception as e:
                    invalid_expressions.append({
                        "text": text,
                        "type": "assignment",
                        "error": str(e),
                    })
            
            # Process arithmetic expressions
            elif len(components) == 3 and components[1] in ['+', '-', '*', '/']:
                try:
                    a = float(components[0]) if components[0].replace('.', '').isdigit() else variables.get(components[0])
                    op = components[1]
                    b = float(components[2]) if components[2].replace('.', '').isdigit() else variables.get(components[2])
                    
                    if a is None or b is None:
                        raise ValueError("Variable not defined")
                    
                    if op == '+':
                        result = a + b
                    elif op == '-':
                        result = a - b
                    elif op == '*':
                        result = a * b
                    elif op == '/':
                        if b == 0:
                            raise ValueError("Division by zero")
                        result = a / b
                    
                    valid_expressions.append({
                        "text": text,
                        "type": "arithmetic",
                        "a": a,
                        "op": op,
                        "b": b,
                        "result": result,
                    })
                except Exception as e:
                    invalid_expressions.append({
                        "text": text,
                        "type": "arithmetic",
                        "error": str(e),
                    })
        
        # Check for final answer and compare with ground truth
        final_answer = None
        answer_match = re.search(r"(?:final answer|result|answer)(?:\s+is)?(?:\s*:)?\s*(\d+(?:\.\d+)?)", reasoning_text, re.IGNORECASE)
        
        if answer_match:
            final_answer = float(answer_match.group(1))
        
        answer_correct = None
        if ground_truth and "answer" in ground_truth and final_answer is not None:
            expected = float(ground_truth["answer"])
            answer_correct = abs(final_answer - expected) < 1e-10
        
        # Determine validation result
        is_valid = len(invalid_expressions) == 0 and (answer_correct is None or answer_correct)
        
        # Confidence based on expressions and answer
        if len(expressions) == 0:
            confidence = 0.5  # No expressions to validate
        else:
            expression_confidence = len(valid_expressions) / len(expressions)
            if answer_correct is not None:
                confidence = (expression_confidence + (1.0 if answer_correct else 0.0)) / 2.0
            else:
                confidence = expression_confidence
        
        details = {
            "valid_expressions": valid_expressions,
            "invalid_expressions": invalid_expressions,
            "final_answer": final_answer,
            "answer_correct": answer_correct,
            "variables": variables,
        }
        
        return ValidationResult(
            result_id=result_id,
            is_valid=is_valid,
            confidence=confidence,
            details=details,
            validation_type="symbolic_math",
        )
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using a simple approach.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # This is a simplified implementation
        # A full implementation would use more sophisticated NLP
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)