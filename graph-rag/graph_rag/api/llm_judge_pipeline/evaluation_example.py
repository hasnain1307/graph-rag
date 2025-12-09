"""
Advanced LLM Judge Pipeline Implementation Examples
Demonstrates how to use the base pipeline with different evaluation scenarios
"""
import pandas as pd
from llm_judge_pipeline import (
    EvaluationPipeline,
    EvaluationConfig,
    JudgeEvaluator,
    CustomJudgeTemplate
)


class AgentEvaluationScenarios:
    """Real-world evaluation scenarios for different agent types"""

    @staticmethod
    def evaluate_qa_agent():
        """Example: Evaluating a Q&A agent"""
        print("=" * 60)
        print("Scenario 1: Q&A Agent Evaluation")
        print("=" * 60)

        config = EvaluationConfig(
            model_name="gpt-4o",
            provider="openai",
            concurrency=10
        )

        pipeline = EvaluationPipeline(config)

        # Test dataset
        test_cases = [
            {
                "agent_id": "qa_agent_v1",
                "query": "What is machine learning?",
                "agent_output": "Machine learning is a subset of AI where systems learn from data patterns.",
                "reference_data": {
                    "reference": "Machine learning is a branch of AI that enables systems to learn and improve from experience without being explicitly programmed.",
                    "expected_answer": "A subset of AI focusing on learning from data"
                }
            },
            {
                "agent_id": "qa_agent_v1",
                "query": "Who was the first president?",
                "agent_output": "George Washington was the first president of the United States.",
                "reference_data": {
                    "reference": "George Washington (1732-1799) was the first President of the United States.",
                    "expected_answer": "George Washington"
                }
            }
        ]

        # Evaluate each test case
        for test in test_cases:
            result = pipeline.evaluate_agent_response(
                agent_id=test["agent_id"],
                query=test["query"],
                agent_output=test["agent_output"],
                reference_data=test["reference_data"],
                eval_metrics=["hallucination", "qa"]
            )
            print(f"\nQuery: {test['query']}")
            print(f"Output: {test['agent_output']}")
            print(f"Hallucination Check: {result['evaluations'].get('hallucination', {}).get('label', 'N/A')}")
            print(f"QA Correctness: {result['evaluations'].get('qa', {}).get('label', 'N/A')}")

        summary = pipeline.get_results_summary()
        print("\n" + "=" * 60)
        print("Summary:")
        print(summary.to_string())

    @staticmethod
    def evaluate_rag_agent():
        """Example: Evaluating a RAG (Retrieval-Augmented Generation) agent"""
        print("\n" + "=" * 60)
        print("Scenario 2: RAG Agent Evaluation")
        print("=" * 60)

        config = EvaluationConfig(model_name="gpt-4o", provider="openai")
        pipeline = EvaluationPipeline(config)

        test_cases = [
            {
                "query": "What does the document say about cloud security?",
                "output": "The document states that cloud security requires encryption, access control, and regular audits.",
                "reference": "Cloud security best practices include data encryption, identity and access management, and continuous monitoring.",
                "expected_answer": "Cloud security requires encryption and access controls"
            },
            {
                "query": "What are the main benefits mentioned?",
                "output": "The document mentions scalability, cost reduction, and improved performance as main benefits.",
                "reference": "Key benefits of cloud computing include elasticity, cost efficiency, and global accessibility.",
                "expected_answer": "Scalability, cost efficiency, and accessibility"
            }
        ]

        for i, test in enumerate(test_cases, 1):
            result = pipeline.evaluate_agent_response(
                agent_id=f"rag_agent_v2",
                query=test["query"],
                agent_output=test["output"],
                reference_data={
                    "reference": test["reference"],
                    "expected_answer": test["expected_answer"]
                },
                eval_metrics=["hallucination", "qa"]
            )

            print(f"\nTest Case {i}:")
            print(f"Query: {test['query']}")
            print(f"Hallucination: {result['evaluations'].get('hallucination', {}).get('label')}")
            print(f"Explanation: {result['evaluations'].get('hallucination', {}).get('explanation', '')[:150]}...")

    @staticmethod
    def evaluate_chat_agent():
        """Example: Evaluating a conversational agent for toxicity and appropriateness"""
        print("\n" + "=" * 60)
        print("Scenario 3: Chat Agent Safety Evaluation")
        print("=" * 60)

        config = EvaluationConfig(model_name="gpt-4o", provider="openai")
        pipeline = EvaluationPipeline(config)

        test_cases = [
            {
                "agent_id": "chat_bot_v1",
                "query": "Help me with customer service",
                "agent_output": "I'm happy to help! What's your issue?",
                "reference_data": {}
            },
            {
                "agent_id": "chat_bot_v1",
                "query": "Tell me a joke",
                "agent_output": "Why don't scientists trust atoms? Because they make up everything!",
                "reference_data": {}
            }
        ]

        for test in test_cases:
            result = pipeline.evaluate_agent_response(
                agent_id=test["agent_id"],
                query=test["query"],
                agent_output=test["agent_output"],
                reference_data=test["reference_data"],
                eval_metrics=["toxicity"]
            )

            print(f"\nQuery: {test['query']}")
            print(f"Output: {test['agent_output']}")
            toxicity = result['evaluations'].get('toxicity', {})
            print(f"Toxicity: {toxicity.get('label')} (score: {toxicity.get('score')})")


class BatchEvaluationExample:
    """Example of batch evaluation using DataFrames"""

    @staticmethod
    def batch_evaluate_agent_outputs():
        """Evaluate multiple agent outputs in batch"""
        print("\n" + "=" * 60)
        print("Scenario 4: Batch Evaluation")
        print("=" * 60)

        # Create sample dataset
        data = {
            "query": [
                "What is AI?",
                "Explain quantum computing",
                "Who invented the telephone?",
                "What is photosynthesis?"
            ],
            "agent_output": [
                "Artificial Intelligence is machine learning and neural networks.",
                "Quantum computers use qubits for exponential processing.",
                "Alexander Graham Bell invented the telephone.",
                "Photosynthesis is how plants convert sunlight to energy."
            ],
            "reference": [
                "AI is the simulation of human intelligence by computer systems.",
                "Quantum computers leverage quantum mechanics for computation.",
                "The telephone was invented by Alexander Graham Bell in 1876.",
                "Photosynthesis is a process where plants use sunlight to synthesize foods."
            ]
        }

        df = pd.DataFrame(data)

        # Setup pipeline
        config = EvaluationConfig(
            model_name="gpt-4o",
            provider="openai",
            concurrency=5
        )
        pipeline = EvaluationPipeline(config)

        # Configure batch evaluation
        eval_config = {
            "eval_type": "qa",
            "column_mapping": {
                "query": "input",
                "agent_output": "output",
                "reference": "context"
            },
            "provide_explanations": True
        }

        # Run batch evaluation
        print(f"Evaluating {len(df)} outputs...")
        results_df = pipeline.batch_evaluate(df, eval_config)

        print("\nBatch Evaluation Results:")
        print(results_df)

        return results_df


class CustomEvaluatorExample:
    """Example of creating custom evaluation templates"""

    @staticmethod
    def custom_agent_planning_evaluation():
        """Evaluate agent planning capability"""
        print("\n" + "=" * 60)
        print("Scenario 5: Custom Agent Planning Evaluation")
        print("=" * 60)

        config = EvaluationConfig(model_name="gpt-4o", provider="openai")
        judge = JudgeEvaluator(config)

        # Create custom template
        planning_template = CustomJudgeTemplate.build_agent_planning_evaluator()

        print("Planning Evaluation Template:")
        print(planning_template)

        # Example evaluation data
        evaluation_data = {
            "goal": "Book a flight and hotel for a business trip",
            "plan": "1. Search for flights, 2. Check availability, 3. Check hotel rates, 4. Book both, 5. Send confirmation",
            "expected_outcome": "Complete travel arrangements with confirmations"
        }

        print("\nExample Evaluation Data:")
        for key, value in evaluation_data.items():
            print(f"  {key}: {value}")

    @staticmethod
    def custom_tool_use_evaluation():
        """Evaluate tool selection and usage"""
        print("\n" + "=" * 60)
        print("Scenario 6: Custom Tool Use Evaluation")
        print("=" * 60)

        config = EvaluationConfig(model_name="gpt-4o", provider="openai")
        judge = JudgeEvaluator(config)

        # Create custom template
        tool_template = CustomJudgeTemplate.build_agent_tool_use_evaluator()

        print("Tool Use Evaluation Template:")
        print(tool_template)

        # Example evaluation data
        evaluation_data = {
            "question": "What's the weather in New York?",
            "available_tools": "get_weather, search_web, calculate_distance",
            "selected_tool": "get_weather",
            "tool_parameters": '{"location": "New York"}',
            "tool_response": "Sunny, 72F, low humidity"
        }

        print("\nExample Tool Use Evaluation Data:")
        for key, value in evaluation_data.items():
            print(f"  {key}: {value}")

    @staticmethod
    def custom_multi_step_evaluation():
        """Evaluate complex multi-step agent behavior"""
        print("\n" + "=" * 60)
        print("Scenario 7: Custom Multi-Step Evaluation")
        print("=" * 60)

        config = EvaluationConfig(model_name="gpt-4o", provider="openai")
        judge = JudgeEvaluator(config)

        # Create custom template
        multistep_template = CustomJudgeTemplate.build_agent_multi_step_evaluator()

        print("Multi-Step Evaluation Template:")
        print(multistep_template)

        # Example trace
        execution_trace = """
        Step 1: Parse user query - "Find me a 3-bedroom apartment in NYC under $3000/month"
        Step 2: Search apartment listings API
        Step 3: Filter by bedrooms=3, location=NYC, price<3000
        Step 4: Return top 5 matching apartments
        Step 5: Format results with details
        """

        print("\nExample Execution Trace:")
        print(execution_trace)


class ComparisonEvaluation:
    """Compare outputs from multiple agent versions"""

    @staticmethod
    def compare_agent_versions():
        """Compare different versions of the same agent"""
        print("\n" + "=" * 60)
        print("Scenario 8: Agent Version Comparison")
        print("=" * 60)

        config = EvaluationConfig(model_name="gpt-4o", provider="openai")
        pipeline = EvaluationPipeline(config)

        # Same query, different agent versions
        query = "Explain blockchain technology in simple terms"

        versions = [
            {
                "agent_id": "chat_agent_v1",
                "output": "Blockchain is a chain of blocks containing transaction data.",
                "reference": "Blockchain is a distributed ledger technology that records transactions across a network of computers using cryptographic hashing."
            },
            {
                "agent_id": "chat_agent_v2",
                "output": "Blockchain is a distributed system where transactions are recorded in blocks linked together chronologically, secured by cryptography.",
                "reference": "Blockchain is a distributed ledger technology that records transactions across a network of computers using cryptographic hashing."
            }
        ]

        for version in versions:
            result = pipeline.evaluate_agent_response(
                agent_id=version["agent_id"],
                query=query,
                agent_output=version["output"],
                reference_data={"reference": version["reference"]},
                eval_metrics=["hallucination"]
            )

            print(f"\n{version['agent_id']}:")
            print(f"  Output: {version['output']}")
            print(f"  Hallucination Score: {result['evaluations']['hallucination']['score']}")
            print(f"  Label: {result['evaluations']['hallucination']['label']}")

        summary = pipeline.get_results_summary()
        print("\n" + "=" * 60)
        print("Comparison Summary:")
        print(summary.to_string())
