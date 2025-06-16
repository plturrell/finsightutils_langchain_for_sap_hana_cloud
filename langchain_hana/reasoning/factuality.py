"""
Factuality measurement framework for questions generated from SAP HANA schemas.

This module provides tools for measuring the factual accuracy of questions and answers
generated from SAP HANA database schemas, tables, and data. It implements a methodology
inspired by the SimpleQA approach for measuring short-form factuality in large language models.
"""

import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import Generation

logger = logging.getLogger(__name__)


class FactualityGrade(str, Enum):
    """Enumeration of factuality grades."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NOT_ATTEMPTED = "not_attempted"
    AMBIGUOUS = "ambiguous"


@dataclass
class SchemaQuestion:
    """Represents a question about a database schema."""
    question_id: str
    question_text: str
    reference_answer: str
    schema_entity: str  # table, column, or relationship
    entity_type: str  # "table", "column", "relationship", "data"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ModelAnswer:
    """Represents a model's answer to a schema question."""
    answer_id: str
    question_id: str
    model_id: str
    answer_text: str
    confidence: Optional[float] = None
    grade: Optional[FactualityGrade] = None
    grading_notes: Optional[str] = None
    response_time: Optional[float] = None
    created_at: float = field(default_factory=time.time)


class SchemaFactualityBenchmark:
    """
    Benchmark for measuring factuality in schema-based questions.
    
    This class provides methods for creating, managing, and evaluating questions
    about database schemas to measure the factual accuracy of language model responses.
    """
    
    def __init__(
        self,
        schema_name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the schema factuality benchmark.
        
        Args:
            schema_name: Name of the database schema being evaluated
            description: Description of the benchmark
            metadata: Additional metadata about the benchmark
        """
        self.benchmark_id = str(uuid.uuid4())
        self.schema_name = schema_name
        self.description = description
        self.metadata = metadata or {}
        self.questions: Dict[str, SchemaQuestion] = {}
        self.model_answers: Dict[str, Dict[str, ModelAnswer]] = {}  # question_id -> model_id -> answer
    
    def add_question(
        self,
        question_text: str,
        reference_answer: str,
        schema_entity: str,
        entity_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a question to the benchmark.
        
        Args:
            question_text: The question text
            reference_answer: The reference answer
            schema_entity: The schema entity (table, column, etc.)
            entity_type: The type of entity ("table", "column", etc.)
            metadata: Additional metadata about the question
            
        Returns:
            The ID of the newly created question
        """
        question_id = str(uuid.uuid4())
        
        question = SchemaQuestion(
            question_id=question_id,
            question_text=question_text,
            reference_answer=reference_answer,
            schema_entity=schema_entity,
            entity_type=entity_type,
            metadata=metadata or {},
        )
        
        self.questions[question_id] = question
        return question_id
    
    def get_question(self, question_id: str) -> Optional[SchemaQuestion]:
        """
        Get a question by ID.
        
        Args:
            question_id: The ID of the question to retrieve
            
        Returns:
            The question or None if not found
        """
        return self.questions.get(question_id)
    
    def get_questions_by_entity(self, schema_entity: str) -> List[SchemaQuestion]:
        """
        Get questions by schema entity.
        
        Args:
            schema_entity: The schema entity to filter by
            
        Returns:
            List of questions for the specified entity
        """
        return [q for q in self.questions.values() if q.schema_entity == schema_entity]
    
    def get_questions_by_type(self, entity_type: str) -> List[SchemaQuestion]:
        """
        Get questions by entity type.
        
        Args:
            entity_type: The entity type to filter by
            
        Returns:
            List of questions for the specified entity type
        """
        return [q for q in self.questions.values() if q.entity_type == entity_type]
    
    def add_model_answer(
        self,
        question_id: str,
        model_id: str,
        answer_text: str,
        confidence: Optional[float] = None,
        response_time: Optional[float] = None,
    ) -> str:
        """
        Add a model's answer to a question.
        
        Args:
            question_id: The ID of the question
            model_id: The ID of the model
            answer_text: The model's answer text
            confidence: The model's stated confidence (0-100)
            response_time: The time taken to generate the answer
            
        Returns:
            The ID of the newly created answer
        """
        if question_id not in self.questions:
            raise ValueError(f"Question ID {question_id} not found")
        
        answer_id = str(uuid.uuid4())
        
        answer = ModelAnswer(
            answer_id=answer_id,
            question_id=question_id,
            model_id=model_id,
            answer_text=answer_text,
            confidence=confidence,
            response_time=response_time,
        )
        
        if question_id not in self.model_answers:
            self.model_answers[question_id] = {}
        
        self.model_answers[question_id][model_id] = answer
        return answer_id
    
    def grade_answer(
        self,
        answer_id: str,
        question_id: str,
        model_id: str,
        grade: FactualityGrade,
        grading_notes: Optional[str] = None,
    ) -> None:
        """
        Grade a model's answer.
        
        Args:
            answer_id: The ID of the answer
            question_id: The ID of the question
            model_id: The ID of the model
            grade: The factuality grade
            grading_notes: Optional notes about the grading
        """
        if question_id not in self.model_answers or model_id not in self.model_answers[question_id]:
            raise ValueError(f"Answer not found for question {question_id} and model {model_id}")
        
        answer = self.model_answers[question_id][model_id]
        if answer.answer_id != answer_id:
            raise ValueError(f"Answer ID mismatch: expected {answer.answer_id}, got {answer_id}")
        
        answer.grade = grade
        answer.grading_notes = grading_notes
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: The ID of the model
            
        Returns:
            Dictionary of performance metrics
        """
        total_questions = len(self.questions)
        answered_questions = 0
        correct_answers = 0
        incorrect_answers = 0
        not_attempted = 0
        ambiguous = 0
        
        # By entity type
        entity_type_metrics = {}
        
        for question_id, question in self.questions.items():
            entity_type = question.entity_type
            
            if entity_type not in entity_type_metrics:
                entity_type_metrics[entity_type] = {
                    "total": 0,
                    "answered": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "not_attempted": 0,
                    "ambiguous": 0,
                }
            
            entity_type_metrics[entity_type]["total"] += 1
            
            if question_id in self.model_answers and model_id in self.model_answers[question_id]:
                answer = self.model_answers[question_id][model_id]
                
                if answer.grade == FactualityGrade.CORRECT:
                    correct_answers += 1
                    entity_type_metrics[entity_type]["correct"] += 1
                    answered_questions += 1
                    entity_type_metrics[entity_type]["answered"] += 1
                elif answer.grade == FactualityGrade.INCORRECT:
                    incorrect_answers += 1
                    entity_type_metrics[entity_type]["incorrect"] += 1
                    answered_questions += 1
                    entity_type_metrics[entity_type]["answered"] += 1
                elif answer.grade == FactualityGrade.NOT_ATTEMPTED:
                    not_attempted += 1
                    entity_type_metrics[entity_type]["not_attempted"] += 1
                elif answer.grade == FactualityGrade.AMBIGUOUS:
                    ambiguous += 1
                    entity_type_metrics[entity_type]["ambiguous"] += 1
        
        # Calculate overall metrics
        overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0
        attempted_accuracy = correct_answers / answered_questions if answered_questions > 0 else 0
        
        # Calculate F-score (harmonic mean of overall accuracy and attempted accuracy)
        if overall_accuracy > 0 and attempted_accuracy > 0:
            f_score = 2 * (overall_accuracy * attempted_accuracy) / (overall_accuracy + attempted_accuracy)
        else:
            f_score = 0
        
        return {
            "model_id": model_id,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "not_attempted": not_attempted,
            "ambiguous": ambiguous,
            "overall_accuracy": overall_accuracy,
            "attempted_accuracy": attempted_accuracy,
            "f_score": f_score,
            "by_entity_type": entity_type_metrics,
        }
    
    def get_confidence_calibration(self, model_id: str) -> Dict[str, Any]:
        """
        Assess the model's confidence calibration.
        
        Args:
            model_id: The ID of the model
            
        Returns:
            Dictionary of calibration metrics
        """
        confidence_bins = {
            "0-10": {"count": 0, "correct": 0},
            "11-20": {"count": 0, "correct": 0},
            "21-30": {"count": 0, "correct": 0},
            "31-40": {"count": 0, "correct": 0},
            "41-50": {"count": 0, "correct": 0},
            "51-60": {"count": 0, "correct": 0},
            "61-70": {"count": 0, "correct": 0},
            "71-80": {"count": 0, "correct": 0},
            "81-90": {"count": 0, "correct": 0},
            "91-100": {"count": 0, "correct": 0},
            "unknown": {"count": 0, "correct": 0},
        }
        
        for question_answers in self.model_answers.values():
            if model_id in question_answers:
                answer = question_answers[model_id]
                
                if answer.confidence is not None:
                    bin_key = None
                    if 0 <= answer.confidence <= 10:
                        bin_key = "0-10"
                    elif 11 <= answer.confidence <= 20:
                        bin_key = "11-20"
                    elif 21 <= answer.confidence <= 30:
                        bin_key = "21-30"
                    elif 31 <= answer.confidence <= 40:
                        bin_key = "31-40"
                    elif 41 <= answer.confidence <= 50:
                        bin_key = "41-50"
                    elif 51 <= answer.confidence <= 60:
                        bin_key = "51-60"
                    elif 61 <= answer.confidence <= 70:
                        bin_key = "61-70"
                    elif 71 <= answer.confidence <= 80:
                        bin_key = "71-80"
                    elif 81 <= answer.confidence <= 90:
                        bin_key = "81-90"
                    elif 91 <= answer.confidence <= 100:
                        bin_key = "91-100"
                else:
                    bin_key = "unknown"
                
                if bin_key:
                    confidence_bins[bin_key]["count"] += 1
                    if answer.grade == FactualityGrade.CORRECT:
                        confidence_bins[bin_key]["correct"] += 1
        
        # Calculate accuracy for each bin
        calibration_metrics = {}
        for bin_key, data in confidence_bins.items():
            if data["count"] > 0:
                accuracy = data["correct"] / data["count"]
            else:
                accuracy = 0
            
            calibration_metrics[bin_key] = {
                "count": data["count"],
                "correct": data["correct"],
                "accuracy": accuracy,
            }
        
        return {
            "model_id": model_id,
            "calibration_by_confidence": calibration_metrics,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the benchmark to a dictionary.
        
        Returns:
            Dictionary representation of the benchmark
        """
        return {
            "benchmark_id": self.benchmark_id,
            "schema_name": self.schema_name,
            "description": self.description,
            "metadata": self.metadata,
            "questions": {qid: asdict(q) for qid, q in self.questions.items()},
            "model_answers": {
                qid: {mid: asdict(a) for mid, a in answers.items()}
                for qid, answers in self.model_answers.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaFactualityBenchmark":
        """
        Create a benchmark from a dictionary.
        
        Args:
            data: Dictionary representation of the benchmark
            
        Returns:
            SchemaFactualityBenchmark instance
        """
        benchmark = cls(
            schema_name=data["schema_name"],
            description=data["description"],
            metadata=data.get("metadata", {}),
        )
        
        benchmark.benchmark_id = data["benchmark_id"]
        
        # Restore questions
        for qid, q_data in data.get("questions", {}).items():
            question = SchemaQuestion(
                question_id=q_data["question_id"],
                question_text=q_data["question_text"],
                reference_answer=q_data["reference_answer"],
                schema_entity=q_data["schema_entity"],
                entity_type=q_data["entity_type"],
                metadata=q_data.get("metadata", {}),
                created_at=q_data.get("created_at", time.time()),
            )
            benchmark.questions[qid] = question
        
        # Restore model answers
        for qid, answers_by_model in data.get("model_answers", {}).items():
            benchmark.model_answers[qid] = {}
            
            for mid, a_data in answers_by_model.items():
                answer = ModelAnswer(
                    answer_id=a_data["answer_id"],
                    question_id=a_data["question_id"],
                    model_id=a_data["model_id"],
                    answer_text=a_data["answer_text"],
                    confidence=a_data.get("confidence"),
                    grade=a_data.get("grade"),
                    grading_notes=a_data.get("grading_notes"),
                    response_time=a_data.get("response_time"),
                    created_at=a_data.get("created_at", time.time()),
                )
                benchmark.model_answers[qid][mid] = answer
        
        return benchmark
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save the benchmark to a JSON file.
        
        Args:
            file_path: Path to the output file
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "SchemaFactualityBenchmark":
        """
        Load a benchmark from a JSON file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            SchemaFactualityBenchmark instance
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return cls.from_dict(data)


class FactualityEvaluator:
    """
    Evaluator for grading the factual accuracy of model answers.
    
    This class provides methods for automatically grading model answers
    using an LLM-based evaluator.
    """
    
    def __init__(
        self,
        grading_model: BaseLanguageModel,
        model_id: str = "factuality-evaluator",
    ):
        """
        Initialize the factuality evaluator.
        
        Args:
            grading_model: Language model for grading answers
            model_id: ID of the grading model
        """
        self.grading_model = grading_model
        self.model_id = model_id
        
        # Define grading prompt template
        self.grading_prompt = PromptTemplate.from_template(
            """You are an expert evaluator assessing the factual accuracy of answers to questions about database schemas.

Question: {question}
Reference Answer: {reference_answer}
Model Answer: {model_answer}

Grade the model's answer using one of the following categories:
- CORRECT: The answer fully contains the reference answer or is semantically equivalent.
- INCORRECT: The answer contradicts the reference answer or contains factual errors.
- NOT_ATTEMPTED: The model declined to answer, expressed uncertainty, or did not provide a clear answer.
- AMBIGUOUS: The answer is partially correct but contains some inaccuracies or is ambiguous.

Explain your reasoning and provide the final grade as a single word: CORRECT, INCORRECT, NOT_ATTEMPTED, or AMBIGUOUS.

Reasoning:
"""
        )
    
    def grade_answer(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
    ) -> Tuple[FactualityGrade, str]:
        """
        Grade a model's answer against a reference answer.
        
        Args:
            question: The question text
            reference_answer: The reference answer
            model_answer: The model's answer
            
        Returns:
            Tuple of (grade, grading_notes)
        """
        # Format the prompt
        prompt = self.grading_prompt.format(
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer,
        )
        
        # Get the grading model's response
        response = self.grading_model.generate([prompt])
        
        # Extract the grade from the response
        grading_text = response.generations[0][0].text.strip()
        grading_notes = grading_text
        
        # Extract the grade from the text
        if "CORRECT" in grading_text.upper():
            grade = FactualityGrade.CORRECT
        elif "INCORRECT" in grading_text.upper():
            grade = FactualityGrade.INCORRECT
        elif "NOT_ATTEMPTED" in grading_text.upper():
            grade = FactualityGrade.NOT_ATTEMPTED
        elif "AMBIGUOUS" in grading_text.upper():
            grade = FactualityGrade.AMBIGUOUS
        else:
            # Default to incorrect if no clear grade is found
            grade = FactualityGrade.INCORRECT
            grading_notes += "\n\nNote: No clear grade was found in the response, defaulting to INCORRECT."
        
        return grade, grading_notes
    
    def grade_benchmark(
        self,
        benchmark: SchemaFactualityBenchmark,
        model_id: str,
    ) -> None:
        """
        Grade all answers for a model in a benchmark.
        
        Args:
            benchmark: The benchmark to grade
            model_id: The ID of the model to grade
        """
        for question_id, question in benchmark.questions.items():
            if question_id in benchmark.model_answers and model_id in benchmark.model_answers[question_id]:
                answer = benchmark.model_answers[question_id][model_id]
                
                # Skip if already graded
                if answer.grade is not None:
                    continue
                
                # Grade the answer
                grade, grading_notes = self.grade_answer(
                    question=question.question_text,
                    reference_answer=question.reference_answer,
                    model_answer=answer.answer_text,
                )
                
                # Update the answer with the grade
                benchmark.grade_answer(
                    answer_id=answer.answer_id,
                    question_id=question_id,
                    model_id=model_id,
                    grade=grade,
                    grading_notes=grading_notes,
                )


class HanaSchemaQuestionGenerator:
    """
    Generator for creating questions about SAP HANA schemas.
    
    This class provides methods for generating questions about database schemas,
    tables, columns, relationships, and data.
    """
    
    def __init__(
        self,
        generation_model: BaseLanguageModel,
        model_id: str = "question-generator",
    ):
        """
        Initialize the question generator.
        
        Args:
            generation_model: Language model for generating questions
            model_id: ID of the generation model
        """
        self.generation_model = generation_model
        self.model_id = model_id
        
        # Define question generation prompt templates
        self.schema_prompt = PromptTemplate.from_template(
            """You are an expert in database design and SAP HANA. 
Generate {num_questions} precise, fact-seeking questions about the following database schema.
Each question should have a single, indisputable answer based on the schema information provided.

Schema Information:
{schema_info}

For each question, provide:
1. A clear, specific question about the schema
2. The definitive, factual answer
3. The specific entity (table, column, relationship) the question refers to
4. The entity type ("table", "column", "relationship")

Example format:
Question: What is the primary key of the CUSTOMER table?
Answer: CUSTOMER_ID
Entity: CUSTOMER.CUSTOMER_ID
Type: column

Your questions should test understanding of:
- Table structures and relationships
- Primary and foreign keys
- Column data types and constraints
- Schema organization and naming conventions

Generate {num_questions} diverse questions:
"""
        )
        
        self.data_prompt = PromptTemplate.from_template(
            """You are an expert in database design and SAP HANA. 
Generate {num_questions} precise, fact-seeking questions about the following database data.
Each question should have a single, indisputable answer based on the data provided.

Schema and Data Information:
{data_info}

For each question, provide:
1. A clear, specific question about the data
2. The definitive, factual answer
3. The specific entity (table, column) the question refers to
4. The entity type (always "data" for these questions)

Example format:
Question: What is the maximum price in the PRODUCTS table?
Answer: 1299.99
Entity: PRODUCTS.PRICE
Type: data

Your questions should test understanding of:
- Data statistics (min, max, average, counts)
- Data distributions and patterns
- Specific data values and relationships
- Data consistency and integrity

Generate {num_questions} diverse questions:
"""
        )
    
    def parse_generated_questions(self, text: str) -> List[Dict[str, str]]:
        """
        Parse generated questions from model output.
        
        Args:
            text: The model's generated text
            
        Returns:
            List of parsed questions
        """
        questions = []
        current_question = {}
        
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            
            if line.startswith("Question:"):
                # Start a new question
                if current_question and "question" in current_question:
                    questions.append(current_question)
                current_question = {"question": line[len("Question:"):].strip()}
            elif line.startswith("Answer:") and current_question:
                current_question["answer"] = line[len("Answer:"):].strip()
            elif line.startswith("Entity:") and current_question:
                current_question["entity"] = line[len("Entity:"):].strip()
            elif line.startswith("Type:") and current_question:
                current_question["type"] = line[len("Type:"):].strip()
        
        # Add the last question
        if current_question and "question" in current_question:
            questions.append(current_question)
        
        return questions
    
    def generate_schema_questions(
        self,
        schema_info: str,
        num_questions: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Generate questions about a database schema.
        
        Args:
            schema_info: Information about the schema
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Format the prompt
        prompt = self.schema_prompt.format(
            schema_info=schema_info,
            num_questions=num_questions,
        )
        
        # Get the model's response
        response = self.generation_model.generate([prompt])
        generated_text = response.generations[0][0].text.strip()
        
        # Parse the generated questions
        return self.parse_generated_questions(generated_text)
    
    def generate_data_questions(
        self,
        data_info: str,
        num_questions: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Generate questions about database data.
        
        Args:
            data_info: Information about the schema and data
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Format the prompt
        prompt = self.data_prompt.format(
            data_info=data_info,
            num_questions=num_questions,
        )
        
        # Get the model's response
        response = self.generation_model.generate([prompt])
        generated_text = response.generations[0][0].text.strip()
        
        # Parse the generated questions
        return self.parse_generated_questions(generated_text)
    
    def create_benchmark_from_schema(
        self,
        schema_name: str,
        schema_info: str,
        num_schema_questions: int = 10,
        data_info: Optional[str] = None,
        num_data_questions: int = 10,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SchemaFactualityBenchmark:
        """
        Create a benchmark from schema information.
        
        Args:
            schema_name: Name of the database schema
            schema_info: Information about the schema
            num_schema_questions: Number of schema questions to generate
            data_info: Optional information about the schema data
            num_data_questions: Number of data questions to generate
            description: Description of the benchmark
            metadata: Additional metadata about the benchmark
            
        Returns:
            SchemaFactualityBenchmark instance
        """
        # Create a new benchmark
        benchmark = SchemaFactualityBenchmark(
            schema_name=schema_name,
            description=description,
            metadata=metadata,
        )
        
        # Generate schema questions
        schema_questions = self.generate_schema_questions(
            schema_info=schema_info,
            num_questions=num_schema_questions,
        )
        
        # Add schema questions to the benchmark
        for q in schema_questions:
            if all(k in q for k in ["question", "answer", "entity", "type"]):
                benchmark.add_question(
                    question_text=q["question"],
                    reference_answer=q["answer"],
                    schema_entity=q["entity"],
                    entity_type=q["type"],
                )
        
        # Generate data questions if data info is provided
        if data_info:
            data_questions = self.generate_data_questions(
                data_info=data_info,
                num_questions=num_data_questions,
            )
            
            # Add data questions to the benchmark
            for q in data_questions:
                if all(k in q for k in ["question", "answer", "entity", "type"]):
                    benchmark.add_question(
                        question_text=q["question"],
                        reference_answer=q["answer"],
                        schema_entity=q["entity"],
                        entity_type=q["type"],
                    )
        
        return benchmark


class AuditStyleEvaluator:
    """
    Evaluator for assessing questions and answers according to audit style guidelines.
    
    This class provides methods for evaluating the quality, clarity, and style of
    questions and answers from both KPMG audit perspective and Jony Ive design perspective.
    """
    
    def __init__(
        self,
        evaluation_model: BaseLanguageModel,
        model_id: str = "audit-style-evaluator",
    ):
        """
        Initialize the audit style evaluator.
        
        Args:
            evaluation_model: Language model for evaluations
            model_id: ID of the evaluation model
        """
        self.evaluation_model = evaluation_model
        self.model_id = model_id
        
        # Define evaluation prompt templates
        self.kpmg_prompt = PromptTemplate.from_template(
            """You are an expert KPMG auditor evaluating database questions and answers for accuracy, precision, and compliance.

Question: {question}
Answer: {answer}
Entity: {entity}
Type: {type}

Evaluate this question-answer pair according to KPMG audit standards, considering:

1. Factual Accuracy: Is the answer objectively correct and verifiable?
2. Precision: Is the question specific and unambiguous?
3. Compliance: Does the question-answer pair align with database auditing best practices?
4. Completeness: Does the answer fully address the question?
5. Traceability: Can the answer be traced back to specific database elements?

Provide a detailed evaluation with a numerical score (1-10) for each criterion and an overall score.
"""
        )
        
        self.ive_prompt = PromptTemplate.from_template(
            """You are Jony Ive, former Chief Design Officer at Apple, evaluating database questions and answers for clarity, elegance, and user experience.

Question: {question}
Answer: {answer}
Entity: {entity}
Type: {type}

Evaluate this question-answer pair according to your design philosophy, considering:

1. Clarity: Is the question expressed in simple, clear language?
2. Elegance: Does the question-answer pair demonstrate elegant thinking?
3. User Experience: Will users find this interaction helpful and intuitive?
4. Simplicity: Does the question-answer pair reduce complexity to its essence?
5. Accessibility: Is the information accessible to both technical and non-technical users?

Provide a detailed evaluation with a numerical score (1-10) for each criterion and an overall score, using your distinctive voice and design perspective.
"""
        )
    
    def evaluate_kpmg_style(
        self,
        question: str,
        answer: str,
        entity: str,
        entity_type: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a question-answer pair according to KPMG audit standards.
        
        Args:
            question: The question text
            answer: The answer text
            entity: The schema entity
            entity_type: The entity type
            
        Returns:
            Dictionary of evaluation results
        """
        # Format the prompt
        prompt = self.kpmg_prompt.format(
            question=question,
            answer=answer,
            entity=entity,
            type=entity_type,
        )
        
        # Get the evaluation model's response
        response = self.evaluation_model.generate([prompt])
        evaluation_text = response.generations[0][0].text.strip()
        
        # Extract scores using a simple heuristic (this could be improved)
        scores = {}
        overall_score = None
        
        for line in evaluation_text.split("\n"):
            line = line.strip()
            
            # Look for lines with "Score:" or similar patterns
            if any(criterion in line.lower() for criterion in ["factual accuracy", "precision", "compliance", "completeness", "traceability"]):
                for criterion in ["factual accuracy", "precision", "compliance", "completeness", "traceability"]:
                    if criterion in line.lower():
                        # Extract the score (1-10)
                        import re
                        score_match = re.search(r'\b([1-9]|10)(?:/10)?\b', line)
                        if score_match:
                            scores[criterion.replace(" ", "_")] = int(score_match.group(1))
            
            # Look for overall score
            if "overall" in line.lower() and "score" in line.lower():
                import re
                overall_match = re.search(r'\b([1-9]|10)(?:/10)?\b', line)
                if overall_match:
                    overall_score = int(overall_match.group(1))
        
        # If no overall score was found, calculate the average
        if overall_score is None and scores:
            overall_score = sum(scores.values()) / len(scores)
        
        return {
            "evaluation_type": "kpmg_audit",
            "scores": scores,
            "overall_score": overall_score,
            "evaluation_text": evaluation_text,
        }
    
    def evaluate_ive_style(
        self,
        question: str,
        answer: str,
        entity: str,
        entity_type: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a question-answer pair according to Jony Ive's design philosophy.
        
        Args:
            question: The question text
            answer: The answer text
            entity: The schema entity
            entity_type: The entity type
            
        Returns:
            Dictionary of evaluation results
        """
        # Format the prompt
        prompt = self.ive_prompt.format(
            question=question,
            answer=answer,
            entity=entity,
            type=entity_type,
        )
        
        # Get the evaluation model's response
        response = self.evaluation_model.generate([prompt])
        evaluation_text = response.generations[0][0].text.strip()
        
        # Extract scores using a simple heuristic (this could be improved)
        scores = {}
        overall_score = None
        
        for line in evaluation_text.split("\n"):
            line = line.strip()
            
            # Look for lines with "Score:" or similar patterns
            if any(criterion in line.lower() for criterion in ["clarity", "elegance", "user experience", "simplicity", "accessibility"]):
                for criterion in ["clarity", "elegance", "user experience", "simplicity", "accessibility"]:
                    if criterion in line.lower():
                        # Extract the score (1-10)
                        import re
                        score_match = re.search(r'\b([1-9]|10)(?:/10)?\b', line)
                        if score_match:
                            scores[criterion.replace(" ", "_")] = int(score_match.group(1))
            
            # Look for overall score
            if "overall" in line.lower() and "score" in line.lower():
                import re
                overall_match = re.search(r'\b([1-9]|10)(?:/10)?\b', line)
                if overall_match:
                    overall_score = int(overall_match.group(1))
        
        # If no overall score was found, calculate the average
        if overall_score is None and scores:
            overall_score = sum(scores.values()) / len(scores)
        
        return {
            "evaluation_type": "ive_design",
            "scores": scores,
            "overall_score": overall_score,
            "evaluation_text": evaluation_text,
        }
    
    def evaluate_benchmark(
        self,
        benchmark: SchemaFactualityBenchmark,
    ) -> Dict[str, Any]:
        """
        Evaluate all questions in a benchmark.
        
        Args:
            benchmark: The benchmark to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        results = {
            "kpmg_evaluations": {},
            "ive_evaluations": {},
            "kpmg_summary": {},
            "ive_summary": {},
        }
        
        kpmg_scores = []
        ive_scores = []
        
        for question_id, question in benchmark.questions.items():
            # KPMG evaluation
            kpmg_eval = self.evaluate_kpmg_style(
                question=question.question_text,
                answer=question.reference_answer,
                entity=question.schema_entity,
                entity_type=question.entity_type,
            )
            results["kpmg_evaluations"][question_id] = kpmg_eval
            
            if kpmg_eval["overall_score"] is not None:
                kpmg_scores.append(kpmg_eval["overall_score"])
            
            # Jony Ive evaluation
            ive_eval = self.evaluate_ive_style(
                question=question.question_text,
                answer=question.reference_answer,
                entity=question.schema_entity,
                entity_type=question.entity_type,
            )
            results["ive_evaluations"][question_id] = ive_eval
            
            if ive_eval["overall_score"] is not None:
                ive_scores.append(ive_eval["overall_score"])
        
        # Calculate summary statistics
        if kpmg_scores:
            results["kpmg_summary"] = {
                "avg_score": sum(kpmg_scores) / len(kpmg_scores),
                "min_score": min(kpmg_scores),
                "max_score": max(kpmg_scores),
                "num_evaluations": len(kpmg_scores),
            }
        
        if ive_scores:
            results["ive_summary"] = {
                "avg_score": sum(ive_scores) / len(ive_scores),
                "min_score": min(ive_scores),
                "max_score": max(ive_scores),
                "num_evaluations": len(ive_scores),
            }
        
        return results