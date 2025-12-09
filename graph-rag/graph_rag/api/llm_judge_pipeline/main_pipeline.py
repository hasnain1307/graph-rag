"""
Base LLM-as-a-Judge Pipeline for Agent Evaluation using Arize Phoenix
This pipeline provides a scalable framework for evaluating other agents using LLM judges.
"""
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass
from issm_api_common.config.settings import config
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    ToxicityEvaluator,
)
from phoenix.evals import llm_classify


@dataclass
class EvaluationConfig:
    """Configuration for the judge evaluation pipeline"""
    model_name: str = "gpt-4o"
    provider: str = "openai"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 20
    timeout: int = 30


class JudgeEvaluator:
    """Base LLM Judge Evaluator for agent outputs"""

    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize the judge evaluator

        Args:
            config: EvaluationConfig object with model settings
        """
        self.config = config or EvaluationConfig()
        self.api_key = self.config.api_key or config.api_key

        # Initialize the LLM for judging
        self.llm = LLM(
            model=self.config.model_name,
            provider=self.config.provider,
            api_key=self.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Initialize pre-built evaluators
        self.hallucination_eval = HallucinationEvaluator(llm=self.llm)
        self.qa_eval = QAEvaluator(llm=self.llm)
        self.relevance_eval = RelevanceEvaluator(llm=self.llm)
        self.toxicity_eval = ToxicityEvaluator(llm=self.llm)

    def evaluate_hallucination(
            self,
            query: str,
            agent_output: str,
            reference_context: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the agent output contains hallucinations

        Args:
            query: The input query to the agent
            agent_output: The output from the agent being evaluated
            reference_context: Ground truth or reference material

        Returns:
            Dictionary with evaluation scores and explanations
        """
        self.hallucination_eval.bind({
            "input": "query",
            "output": "agent_output",
            "context": "reference_context"
        })

        result = self.hallucination_eval.evaluate({
            "query": query,
            "agent_output": agent_output,
            "reference_context": reference_context
        })

        return {
            "metric": "hallucination",
            "score": result[0].score,
            "label": result[0].label,
            "explanation": result[0].explanation,
            "metadata": result[0].metadata
        }

    def evaluate_qa_correctness(
            self,
            question: str,
            agent_answer: str,
            reference_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if agent answer correctly answers the question

        Args:
            question: The question posed to the agent
            agent_answer: The agent's response
            reference_answer: The correct/expected answer

        Returns:
            Dictionary with evaluation scores and explanations
        """
        self.qa_eval.bind({
            "input": "question",
            "output": "agent_answer",
            "context": "reference_answer"
        })

        result = self.qa_eval.evaluate({
            "question": question,
            "agent_answer": agent_answer,
            "reference_answer": reference_answer
        })

        return {
            "metric": "qa_correctness",
            "score": result[0].score,
            "label": result[0].label,
            "explanation": result[0].explanation,
            "metadata": result[0].metadata
        }

    def evaluate_toxicity(self, text: str) -> Dict[str, Any]:
        """
        Evaluate if agent output contains toxic content

        Args:
            text: Text to evaluate for toxicity

        Returns:
            Dictionary with toxicity evaluation results
        """
        self.toxicity_eval.bind({"input": "text"})
        result = self.toxicity_eval.evaluate({"text": text})

        return {
            "metric": "toxicity",
            "score": result[0].score,
            "label": result[0].label,
            "explanation": result[0].explanation if hasattr(result[0], 'explanation') else None,
            "metadata": result[0].metadata
        }

    def evaluate_batch(
            self,
            dataframe: pd.DataFrame,
            eval_type: str,
            column_mapping: Dict[str, str],
            provide_explanations: bool = True
    ) -> pd.DataFrame:
        """
        Run batch evaluations on multiple agent outputs

        Args:
            dataframe: DataFrame containing agent outputs and reference data
            eval_type: Type of evaluation ("hallucination", "qa", "toxicity", "relevance")
            column_mapping: Map DataFrame columns to evaluation input names
            provide_explanations: Whether to include explanations in results

        Returns:
            DataFrame with evaluation results
        """
        if eval_type == "hallucination":
            evaluator = self.hallucination_eval
            template = None  # Uses default template
        elif eval_type == "qa":
            evaluator = self.qa_eval
            template = None
        elif eval_type == "toxicity":
            evaluator = self.toxicity_eval
            template = None
        elif eval_type == "relevance":
            evaluator = self.relevance_eval
            template = None
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")

        # Rename columns according to mapping
        df_mapped = dataframe.rename(columns=column_mapping)

        # Run batch classification
        results = llm_classify(
            dataframe=df_mapped,
            model=self.llm,
            concurrency=self.config.concurrency,
            provide_explanation=provide_explanations
        )

        return results


class CustomJudgeTemplate:
    """Create custom evaluation templates for domain-specific judging"""

    def __init__(self, llm: LLM):
        self.llm = llm

    def create_custom_evaluator(
            self,
            name: str,
            prompt_template: str,
            rails: List[str],
            provide_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Create a custom evaluation template

        Args:
            name: Name of the evaluator
            prompt_template: The evaluation prompt with {input}, {output}, etc.
            rails: List of valid output labels (e.g., ["correct", "incorrect"])
            provide_explanation: Whether judge should explain its decision

        Returns:
            Evaluator configuration dict
        """
        return {
            "name": name,
            "template": prompt_template,
            "rails": rails,
            "provide_explanation": provide_explanation,
            "llm": self.llm
        }

    @staticmethod
    def build_agent_planning_evaluator() -> str:
        """Build an evaluator for agent planning and reasoning"""
        return """
You are an expert at evaluating AI agent planning and reasoning.

Task:
Evaluate whether the agent's plan is logically sound and efficiently reaches the goal.

Context:
- Agent Goal: {goal}
- Agent Plan: {plan}
- Expected Outcome: {expected_outcome}

Evaluate the plan on:
1. Logical Correctness: Does the plan logically lead to the goal?
2. Efficiency: Could the plan be executed with fewer steps?
3. Safety: Are there any potential risks or violations?
4. Completeness: Does the plan address all requirements?

Provide your assessment as one of: [correct, incomplete, inefficient, unsafe]
"""

    @staticmethod
    def build_agent_tool_use_evaluator() -> str:
        """Build an evaluator for agent tool selection and usage"""
        return """
You are an expert at evaluating AI agent tool usage and selection.

Task:
Determine if the agent selected the right tool and used it correctly.

Context:
- User Question: {question}
- Available Tools: {available_tools}
- Tool Selected: {selected_tool}
- Tool Parameters: {tool_parameters}
- Tool Response: {tool_response}

Evaluate whether:
1. The correct tool was selected for this task
2. The parameters are appropriate
3. The tool was used correctly

Provide your assessment as one of: [correct, wrong_tool, wrong_params, misused]
"""

    @staticmethod
    def build_agent_multi_step_evaluator() -> str:
        """Build an evaluator for multi-step agent behavior"""
        return """
You are an expert at evaluating multi-step AI agent behavior.

Task:
Assess the quality of the agent's multi-step reasoning and execution.

Execution Trace:
{execution_trace}

Final Output: {final_output}
Expected Output: {expected_output}

Evaluate:
1. Step Coherence: Does each step logically follow?
2. Goal Alignment: Does the sequence move toward the goal?
3. Error Recovery: How well did the agent handle errors?
4. Output Quality: Does the final output meet requirements?

Rate as: [excellent, good, adequate, poor]
"""


class EvaluationPipeline:
    """Complete evaluation pipeline for agent evaluation"""

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.judge = JudgeEvaluator(self.config)
        self.results = []

    def evaluate_agent_response(
            self,
            agent_id: str,
            query: str,
            agent_output: str,
            reference_data: Dict[str, str],
            eval_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single agent response against multiple metrics

        Args:
            agent_id: Identifier for the agent being evaluated
            query: The input query
            agent_output: The agent's response
            reference_data: Reference data for evaluation (context, expected answer, etc.)
            eval_metrics: List of metrics to use (hallucination, qa, toxicity, etc.)

        Returns:
            Comprehensive evaluation results
        """
        if eval_metrics is None:
            eval_metrics = ["hallucination", "qa", "toxicity"]

        results = {
            "agent_id": agent_id,
            "query": query,
            "agent_output": agent_output,
            "evaluations": {}
        }

        # Run requested evaluations
        for metric in eval_metrics:
            try:
                if metric == "hallucination" and "reference" in reference_data:
                    results["evaluations"]["hallucination"] = self.judge.evaluate_hallucination(
                        query=query,
                        agent_output=agent_output,
                        reference_context=reference_data.get("reference", "")
                    )

                elif metric == "qa" and "expected_answer" in reference_data:
                    results["evaluations"]["qa"] = self.judge.evaluate_qa_correctness(
                        question=query,
                        agent_answer=agent_output,
                        reference_answer=reference_data.get("expected_answer", "")
                    )

                elif metric == "toxicity":
                    results["evaluations"]["toxicity"] = self.judge.evaluate_toxicity(
                        agent_output
                    )
            except Exception as e:
                results["evaluations"][metric] = {"error": str(e)}

        self.results.append(results)
        return results

    def batch_evaluate(
            self,
            dataframe: pd.DataFrame,
            eval_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Evaluate a batch of agent outputs

        Args:
            dataframe: DataFrame with columns: agent_id, query, output, reference, expected_answer
            eval_config: Dict with 'eval_type' and 'column_mapping' keys

        Returns:
            DataFrame with evaluation results
        """
        return self.judge.evaluate_batch(
            dataframe=dataframe,
            eval_type=eval_config.get("eval_type", "hallucination"),
            column_mapping=eval_config.get("column_mapping", {}),
            provide_explanations=eval_config.get("provide_explanations", True)
        )

    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all evaluations run in this pipeline"""
        summary_data = []

        for result in self.results:
            row = {
                "agent_id": result["agent_id"],
                "query": result["query"][:100],  # Truncate for display
                "num_evaluations": len(result["evaluations"])
            }

            # Add scores for each metric
            for metric, eval_result in result["evaluations"].items():
                if "score" in eval_result:
                    row[f"{metric}_score"] = eval_result["score"]
                    row[f"{metric}_label"] = eval_result.get("label", "N/A")

            summary_data.append(row)

        return pd.DataFrame(summary_data)
