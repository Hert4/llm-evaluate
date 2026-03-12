"""
Ground Truth Generator using OpenAI-compatible API
"""
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..config import OpenAIConfig
from ..data_parsers.base import EvalSample
from .prompts import GTPromptTemplates

logger = logging.getLogger(__name__)


@dataclass
class GTResult:
    """Result of ground truth generation"""
    sample_id: str
    ground_truth: str
    success: bool
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


class GroundTruthGenerator:
    """
    Generate ground truth for evaluation samples using OpenAI-compatible API.
    """

    def __init__(self, config: OpenAIConfig):
        """
        Initialize generator.

        Args:
            config: OpenAI-compatible API configuration
        """
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy initialize OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
        return self._client

    def _get_async_client(self):
        """Get async OpenAI client"""
        try:
            from openai import AsyncOpenAI
            return AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

    def generate_single(
        self,
        sample: EvalSample,
        task_type: str = "qa",
        custom_prompt: Optional[str] = None,
    ) -> GTResult:
        """
        Generate ground truth for a single sample.

        Args:
            sample: Evaluation sample
            task_type: Type of task (qa, summarization, translation, etc.)
            custom_prompt: Custom prompt template (optional)

        Returns:
            GTResult with generated ground truth
        """
        try:
            # Build prompt
            if custom_prompt:
                prompt = custom_prompt.format(
                    input=sample.input,
                    context=sample.context or "",
                    **sample.metadata
                )
            else:
                prompt = GTPromptTemplates.get_prompt_for_task(
                    task_type=task_type,
                    input_text=sample.input,
                    context=sample.context,
                    choices=sample.choices,
                    tools=json.dumps(sample.expected_tool_calls) if sample.expected_tool_calls else "",
                )

            # Call API
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
            )

            ground_truth = response.choices[0].message.content.strip()

            return GTResult(
                sample_id=sample.id,
                ground_truth=ground_truth,
                success=True,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )

        except Exception as e:
            logger.error(f"Error generating GT for sample {sample.id}: {e}")
            return GTResult(
                sample_id=sample.id,
                ground_truth="",
                success=False,
                error=str(e),
            )

    async def _generate_single_async(
        self,
        client,
        sample: EvalSample,
        task_type: str,
        custom_prompt: Optional[str],
        semaphore: asyncio.Semaphore,
    ) -> GTResult:
        """Async single generation with semaphore for rate limiting"""
        async with semaphore:
            try:
                if custom_prompt:
                    prompt = custom_prompt.format(
                        input=sample.input,
                        context=sample.context or "",
                        **sample.metadata
                    )
                else:
                    prompt = GTPromptTemplates.get_prompt_for_task(
                        task_type=task_type,
                        input_text=sample.input,
                        context=sample.context,
                        choices=sample.choices,
                        tools=json.dumps(sample.expected_tool_calls) if sample.expected_tool_calls else "",
                    )

                response = await client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_tokens,
                )

                ground_truth = response.choices[0].message.content.strip()

                return GTResult(
                    sample_id=sample.id,
                    ground_truth=ground_truth,
                    success=True,
                )

            except Exception as e:
                logger.error(f"Error generating GT for sample {sample.id}: {e}")
                return GTResult(
                    sample_id=sample.id,
                    ground_truth="",
                    success=False,
                    error=str(e),
                )

    async def generate_batch_async(
        self,
        samples: List[EvalSample],
        task_type: str = "qa",
        custom_prompt: Optional[str] = None,
        max_concurrent: int = 10,
    ) -> List[GTResult]:
        """
        Generate ground truth for multiple samples asynchronously.

        Args:
            samples: List of evaluation samples
            task_type: Type of task
            custom_prompt: Custom prompt template
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of GTResult
        """
        client = self._get_async_client()
        semaphore = asyncio.Semaphore(max_concurrent)

        tasks = [
            self._generate_single_async(client, sample, task_type, custom_prompt, semaphore)
            for sample in samples
        ]

        results = await asyncio.gather(*tasks)
        await client.close()
        return results

    def generate_batch(
        self,
        samples: List[EvalSample],
        task_type: str = "qa",
        custom_prompt: Optional[str] = None,
        max_concurrent: int = 10,
    ) -> List[GTResult]:
        """
        Generate ground truth for multiple samples (sync wrapper for async).

        Args:
            samples: List of evaluation samples
            task_type: Type of task
            custom_prompt: Custom prompt template
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of GTResult
        """
        return asyncio.run(
            self.generate_batch_async(samples, task_type, custom_prompt, max_concurrent)
        )

    def update_samples_with_gt(
        self,
        samples: List[EvalSample],
        results: List[GTResult],
    ) -> List[EvalSample]:
        """
        Update samples with generated ground truth.

        Args:
            samples: Original samples
            results: Generated ground truth results

        Returns:
            Updated samples with reference field filled
        """
        results_map = {r.sample_id: r for r in results}

        updated_samples = []
        for sample in samples:
            result = results_map.get(sample.id)
            if result and result.success:
                sample.reference = result.ground_truth
            updated_samples.append(sample)

        return updated_samples
