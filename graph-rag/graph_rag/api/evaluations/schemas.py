"""
Pydantic models for LLM Judge evaluation requests and responses
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum


class EvaluationMetricEnum(str, Enum):
    """Available evaluation metrics"""
    HALLUCINATION = "hallucination"
    QA = "qa"
    TOXICITY = "toxicity"
    RELEVANCE = "relevance"


class SingleEvaluationRequest(BaseModel):
    """Request model for single agent response evaluation"""

    agent_id: str = Field(..., description="Unique identifier for the agent being evaluated")
    query: str = Field(..., description="The input query to the agent")
    agent_output: str = Field(..., description="The output/response from the agent")
    eval_metrics: List[EvaluationMetricEnum] = Field(
        default=[EvaluationMetricEnum.HALLUCINATION, EvaluationMetricEnum.TOXICITY],
        description="List of metrics to evaluate"
    )
    reference_data: Optional[Dict[str, str]] = Field(
        default=None,
        description="Reference data for evaluation (reference, expected_answer, etc.)"
    )

    class Config:
        schema_extra = {
            "example": {
                "agent_id": "my_agent_v1",
                "query": "What is machine learning?",
                "agent_output": "Machine learning is a type of AI that learns from data.",
                "eval_metrics": ["hallucination", "qa"],
                "reference_data": {
                    "reference": "Machine learning is a subset of AI...",
                    "expected_answer": "A subset of AI"
                }
            }
        }


class HallucinationEvaluationRequest(BaseModel):
    """Request model for hallucination evaluation"""

    query: str = Field(..., description="The input query")
    agent_output: str = Field(..., description="The agent's response")
    reference_context: str = Field(..., description="Reference/ground truth context")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "agent_output": "Paris is the capital of France.",
                "reference_context": "Paris is the capital and largest city of France."
            }
        }


class QAEvaluationRequest(BaseModel):
    """Request model for Q&A correctness evaluation"""

    question: str = Field(..., description="The question posed")
    agent_answer: str = Field(..., description="The agent's answer")
    reference_answer: str = Field(..., description="The correct/expected answer")

    class Config:
        schema_extra = {
            "example": {
                "question": "Who invented Python?",
                "agent_answer": "Guido van Rossum",
                "reference_answer": "Python was created by Guido van Rossum in 1989"
            }
        }


class ToxicityEvaluationRequest(BaseModel):
    """Request model for toxicity evaluation"""

    text: str = Field(..., description="Text to evaluate for toxicity")

    class Config:
        schema_extra = {
            "example": {
                "text": "This is a helpful and respectful response."
            }
        }


class EvaluationScoreResponse(BaseModel):
    """Response model for evaluation results"""

    metric: str = Field(..., description="The evaluation metric used")
    score: float = Field(..., description="Evaluation score (0-1)")
    label: str = Field(..., description="Label/classification result")
    explanation: Optional[str] = Field(None, description="Explanation for the evaluation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "metric": "hallucination",
                "score": 0.95,
                "label": "factual",
                "explanation": "The response correctly identifies Paris as the capital of France.",
                "metadata": {"model": "gpt-4o"}
            }
        }


class SingleEvaluationResponse(BaseModel):
    """Response model for single agent evaluation"""

    agent_id: str = Field(..., description="Agent identifier")
    query: str = Field(..., description="The input query")
    agent_output: str = Field(..., description="The agent's output")
    evaluations: Dict[str, EvaluationScoreResponse] = Field(
        ...,
        description="Dictionary of evaluation results by metric"
    )

    class Config:
        schema_extra = {
            "example": {
                "agent_id": "my_agent_v1",
                "query": "What is machine learning?",
                "agent_output": "Machine learning is a type of AI...",
                "evaluations": {
                    "hallucination": {
                        "metric": "hallucination",
                        "score": 0.9,
                        "label": "factual",
                        "explanation": "..."
                    },
                    "qa": {
                        "metric": "qa",
                        "score": 0.85,
                        "label": "correct",
                        "explanation": "..."
                    }
                }
            }
        }


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation"""

    eval_type: EvaluationMetricEnum = Field(
        ...,
        description="Type of evaluation to run on batch"
    )
    data: List[Dict[str, str]] = Field(
        ...,
        description="List of evaluation items with required fields"
    )
    column_mapping: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of DataFrame columns to evaluation input names"
    )
    provide_explanations: bool = Field(
        True,
        description="Whether to include explanations in results"
    )

    class Config:
        schema_extra = {
            "example": {
                "eval_type": "hallucination",
                "data": [
                    {
                        "query": "What is AI?",
                        "agent_output": "AI is artificial intelligence.",
                        "reference": "AI is the simulation of human intelligence."
                    }
                ],
                "column_mapping": {
                    "query": "input",
                    "agent_output": "output",
                    "reference": "context"
                },
                "provide_explanations": True
            }
        }


class BatchEvaluationResponse(BaseModel):
    """Response model for batch evaluation"""

    total_evaluated: int = Field(..., description="Total items evaluated")
    eval_type: str = Field(..., description="Type of evaluation performed")
    results: List[Dict[str, Any]] = Field(..., description="Evaluation results")
    summary_stats: Optional[Dict[str, float]] = Field(
        None,
        description="Summary statistics (avg score, pass rate, etc.)"
    )


class HealthCheckResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    service: str = Field(..., description="Service name")


class ErrorResponse(BaseModel):
    """Response model for errors"""

    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    code: str = Field(default="INTERNAL_ERROR", description="Error code")