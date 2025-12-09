"""
Service layer for LLM Judge evaluations
Handles business logic for evaluating agent outputs
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from abc import ABC, abstractmethod

from llm_as_judge.api.llm_judge_pipeline.main_pipeline import (
    EvaluationPipeline,
    EvaluationConfig,
    JudgeEvaluator,
)
from llm_as_judge.api.evaluations.schemas import (
    SingleEvaluationRequest,
    SingleEvaluationResponse,
    EvaluationScoreResponse,
    HallucinationEvaluationRequest,
    QAEvaluationRequest,
    ToxicityEvaluationRequest,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
)

logger = logging.getLogger(__name__)


class IEvaluationService(ABC):
    """Interface for evaluation service"""

    @abstractmethod
    def evaluate_agent_response(self, request: SingleEvaluationRequest) -> SingleEvaluationResponse:
        """Evaluate a single agent response"""
        pass

    @abstractmethod
    def evaluate_hallucination(self, request: HallucinationEvaluationRequest) -> EvaluationScoreResponse:
        """Evaluate hallucination in a response"""
        pass

    @abstractmethod
    def evaluate_qa(self, request: QAEvaluationRequest) -> EvaluationScoreResponse:
        """Evaluate Q&A correctness"""
        pass

    @abstractmethod
    def evaluate_toxicity(self, request: ToxicityEvaluationRequest) -> EvaluationScoreResponse:
        """Evaluate toxicity in text"""
        pass

    @abstractmethod
    def batch_evaluate(self, request: BatchEvaluationRequest) -> BatchEvaluationResponse:
        """Batch evaluate multiple outputs"""
        pass


class EvaluationService(IEvaluationService):
    """
    Service for LLM-based agent evaluation
    Wraps the EvaluationPipeline and provides business logic
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the evaluation service

        Args:
            config: EvaluationConfig with model and concurrency settings
        """
        self.config = config or EvaluationConfig(
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            concurrency=10
        )

        self.pipeline = EvaluationPipeline(self.config)
        self.judge = JudgeEvaluator(self.config)

        logger.info(f"EvaluationService initialized with config: {self.config}")

    def evaluate_agent_response(
            self,
            request: SingleEvaluationRequest
    ) -> SingleEvaluationResponse:
        """
        Evaluate a single agent response across multiple metrics

        Args:
            request: SingleEvaluationRequest with agent output and metrics

        Returns:
            SingleEvaluationResponse with evaluation results

        Raises:
            ValueError: If invalid metric requested
            Exception: If evaluation fails
        """
        try:
            logger.info(
                f"Evaluating agent {request.agent_id} with metrics: {request.eval_metrics}"
            )

            # Convert enum to string list
            metrics = [m.value for m in request.eval_metrics]

            # Use pipeline to evaluate
            result = self.pipeline.evaluate_agent_response(
                agent_id=request.agent_id,
                query=request.query,
                agent_output=request.agent_output,
                reference_data=request.reference_data or {},
                eval_metrics=metrics
            )

            # Convert results to response model
            evaluations = {}
            for metric_name, metric_result in result.get("evaluations", {}).items():
                if "error" not in metric_result:
                    evaluations[metric_name] = EvaluationScoreResponse(
                        metric=metric_result.get("metric", metric_name),
                        score=metric_result.get("score", 0.0),
                        label=metric_result.get("label", "unknown"),
                        explanation=metric_result.get("explanation"),
                        metadata=metric_result.get("metadata")
                    )
                else:
                    logger.warning(
                        f"Error evaluating {metric_name} for agent {request.agent_id}: "
                        f"{metric_result['error']}"
                    )

            response = SingleEvaluationResponse(
                agent_id=request.agent_id,
                query=request.query,
                agent_output=request.agent_output,
                evaluations=evaluations
            )

            logger.info(f"Successfully evaluated agent {request.agent_id}")
            return response

        except Exception as e:
            logger.error(f"Error evaluating agent response: {str(e)}", exc_info=True)
            raise

    def evaluate_hallucination(
            self,
            request: HallucinationEvaluationRequest
    ) -> EvaluationScoreResponse:
        """
        Evaluate if output contains hallucinations

        Args:
            request: HallucinationEvaluationRequest with query, output, and reference

        Returns:
            EvaluationScoreResponse with hallucination score
        """
        try:
            logger.info("Evaluating hallucination")

            result = self.judge.evaluate_hallucination(
                query=request.query,
                agent_output=request.agent_output,
                reference_context=request.reference_context
            )

            response = EvaluationScoreResponse(
                metric=result.get("metric", "hallucination"),
                score=result.get("score", 0.0),
                label=result.get("label", "unknown"),
                explanation=result.get("explanation"),
                metadata=result.get("metadata")
            )

            logger.info(f"Hallucination evaluation: {response.label}")
            return response

        except Exception as e:
            logger.error(f"Error evaluating hallucination: {str(e)}", exc_info=True)
            raise

    def evaluate_qa(
            self,
            request: QAEvaluationRequest
    ) -> EvaluationScoreResponse:
        """
        Evaluate Q&A correctness

        Args:
            request: QAEvaluationRequest with question, answer, and reference

        Returns:
            EvaluationScoreResponse with QA correctness score
        """
        try:
            logger.info("Evaluating Q&A correctness")

            result = self.judge.evaluate_qa_correctness(
                question=request.question,
                agent_answer=request.agent_answer,
                reference_answer=request.reference_answer
            )

            response = EvaluationScoreResponse(
                metric=result.get("metric", "qa_correctness"),
                score=result.get("score", 0.0),
                label=result.get("label", "unknown"),
                explanation=result.get("explanation"),
                metadata=result.get("metadata")
            )

            logger.info(f"Q&A evaluation: {response.label}")
            return response

        except Exception as e:
            logger.error(f"Error evaluating Q&A: {str(e)}", exc_info=True)
            raise

    def evaluate_toxicity(
            self,
            request: ToxicityEvaluationRequest
    ) -> EvaluationScoreResponse:
        """
        Evaluate toxicity in text

        Args:
            request: ToxicityEvaluationRequest with text

        Returns:
            EvaluationScoreResponse with toxicity score
        """
        try:
            logger.info("Evaluating toxicity")

            result = self.judge.evaluate_toxicity(
                text=request.text
            )

            response = EvaluationScoreResponse(
                metric=result.get("metric", "toxicity"),
                score=result.get("score", 0.0),
                label=result.get("label", "unknown"),
                explanation=result.get("explanation"),
                metadata=result.get("metadata")
            )

            logger.info(f"Toxicity evaluation: {response.label}")
            return response

        except Exception as e:
            logger.error(f"Error evaluating toxicity: {str(e)}", exc_info=True)
            raise

    def batch_evaluate(
            self,
            request: BatchEvaluationRequest
    ) -> BatchEvaluationResponse:
        """
        Batch evaluate multiple outputs

        Args:
            request: BatchEvaluationRequest with data and eval type

        Returns:
            BatchEvaluationResponse with batch results
        """
        try:
            logger.info(f"Starting batch evaluation of {len(request.data)} items")

            # Convert list of dicts to DataFrame
            df = pd.DataFrame(request.data)

            eval_config = {
                "eval_type": request.eval_type.value,
                "column_mapping": request.column_mapping or {},
                "provide_explanations": request.provide_explanations
            }

            # Run batch evaluation
            results_df = self.pipeline.batch_evaluate(df, eval_config)

            # Convert results to list of dicts
            results_list = results_df.to_dict(orient="records")

            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(results_df)

            response = BatchEvaluationResponse(
                total_evaluated=len(results_list),
                eval_type=request.eval_type.value,
                results=results_list,
                summary_stats=summary_stats
            )

            logger.info(
                f"Batch evaluation completed: {len(results_list)} items, "
                f"avg score: {summary_stats.get('avg_score', 'N/A')}"
            )
            return response

        except Exception as e:
            logger.error(f"Error in batch evaluation: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def _calculate_summary_stats(results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary statistics from evaluation results

        Args:
            results_df: DataFrame with evaluation results

        Returns:
            Dictionary with summary statistics
        """
        stats = {}

        # Find numeric columns (scores)
        numeric_cols = results_df.select_dtypes(include=['float64', 'float32']).columns

        if len(numeric_cols) > 0:
            # Average score across all numeric columns
            avg_score = results_df[numeric_cols].mean().mean()
            stats["avg_score"] = float(avg_score)

            # Min and max scores
            stats["min_score"] = float(results_df[numeric_cols].min().min())
            stats["max_score"] = float(results_df[numeric_cols].max().max())

            # Count of "passing" results (score > 0.7)
            passing = (results_df[numeric_cols] > 0.7).sum().sum()
            total = results_df[numeric_cols].size
            stats["pass_rate"] = float(passing / total) if total > 0 else 0.0

        return stats

    def get_service_config(self) -> Dict[str, Any]:
        """
        Get current service configuration

        Returns:
            Dictionary with service configuration
        """
        return {
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "concurrency": self.config.concurrency,
            "timeout": self.config.timeout
        }


class EvaluationServiceFactory:
    """Factory for creating evaluation service instances"""

    _instance: Optional[EvaluationService] = None

    @classmethod
    def get_service(
            cls,
            config: Optional[EvaluationConfig] = None,
            force_new: bool = False
    ) -> EvaluationService:
        """
        Get or create an evaluation service instance (singleton pattern)

        Args:
            config: Optional EvaluationConfig
            force_new: If True, create new instance instead of returning cached

        Returns:
            EvaluationService instance
        """
        if force_new or cls._instance is None:
            logger.info("Creating new EvaluationService instance")
            cls._instance = EvaluationService(config)

        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance"""
        cls._instance = None
        logger.info("EvaluationService instance reset")