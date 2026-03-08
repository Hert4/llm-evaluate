"""
Safety Metrics: FactScore, IFEval
"""
from typing import List, Dict, Any, Optional
import re

from .base import BaseMetric, LLMBasedMetric, MetricResult
from ..data_parsers.base import EvalSample
from ..config import OpenAIConfig


class FactScoreMetric(LLMBasedMetric):
    """
    FactScore: Measures factual accuracy by breaking down response into atomic facts.
    """

    name = "factscore"

    FACT_EXTRACTION_PROMPT = """Break down the following text into atomic facts. Each fact should be a single, verifiable statement.

Text: {text}

List each atomic fact on a separate line, numbered:
1.
2.
..."""

    FACT_VERIFICATION_PROMPT = """Verify if the following fact is correct.

Fact: {fact}

{reference_section}

Answer "CORRECT", "INCORRECT", or "UNVERIFIABLE"."""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(openai_config, config)

    def _extract_facts(self, text: str) -> List[str]:
        """Extract atomic facts from text"""
        prompt = self.FACT_EXTRACTION_PROMPT.format(text=text)
        response = self._call_llm(prompt)

        # Parse numbered list
        facts = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.?\s*', line):
                fact = re.sub(r'^\d+\.?\s*', '', line).strip()
                if fact:
                    facts.append(fact)
        return facts

    def _verify_fact(self, fact: str, reference: Optional[str] = None) -> str:
        """Verify a single fact"""
        reference_section = f"Reference information:\n{reference}" if reference else ""
        prompt = self.FACT_VERIFICATION_PROMPT.format(
            fact=fact,
            reference_section=reference_section
        )
        response = self._call_llm(prompt)

        if "INCORRECT" in response.upper():
            return "incorrect"
        elif "CORRECT" in response.upper():
            return "correct"
        else:
            return "unverifiable"

    def _compute_single(self, sample: EvalSample) -> Dict[str, Any]:
        """Compute FactScore for single sample"""
        facts = self._extract_facts(sample.output)

        if not facts:
            return {"score": 1.0, "num_facts": 0, "correct": 0, "incorrect": 0}

        correct = 0
        incorrect = 0
        unverifiable = 0

        for fact in facts:
            result = self._verify_fact(fact, sample.reference)
            if result == "correct":
                correct += 1
            elif result == "incorrect":
                incorrect += 1
            else:
                unverifiable += 1

        # Score = correct / (correct + incorrect), excluding unverifiable
        verifiable = correct + incorrect
        score = correct / verifiable if verifiable > 0 else 1.0

        return {
            "score": score,
            "num_facts": len(facts),
            "correct": correct,
            "incorrect": incorrect,
            "unverifiable": unverifiable,
        }

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute FactScore for all samples"""
        if not samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples provided"},
            )

        per_sample_scores = []
        total_facts = 0
        total_correct = 0
        total_incorrect = 0

        for sample in samples:
            result = self._compute_single(sample)
            per_sample_scores.append(result["score"])
            total_facts += result["num_facts"]
            total_correct += result["correct"]
            total_incorrect += result["incorrect"]

        avg_score = sum(per_sample_scores) / len(per_sample_scores)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={
                "total_facts": total_facts,
                "total_correct": total_correct,
                "total_incorrect": total_incorrect,
                "num_samples": len(samples),
            },
            per_sample_scores=per_sample_scores,
        )


class IFEvalMetric(BaseMetric):
    """
    IFEval: Instruction Following Evaluation.
    Measures if the model follows specific instructions like word count, format, etc.
    """

    name = "ifeval"

    # Common instruction patterns and their validators
    INSTRUCTION_PATTERNS = {
        "word_count": r"(?:exactly|write|use)\s+(\d+)\s+words?",
        "sentence_count": r"(?:exactly|write|use)\s+(\d+)\s+sentences?",
        "no_word": r"(?:do not|don't|avoid|without)\s+(?:use|using|the word)\s+['\"]?(\w+)['\"]?",
        "must_include": r"(?:must|should)\s+include\s+['\"]?(.+?)['\"]?(?:\s|$|\.)",
        "start_with": r"(?:start|begin)\s+with\s+['\"]?(.+?)['\"]?(?:\s|$|\.)",
        "end_with": r"(?:end|finish)\s+with\s+['\"]?(.+?)['\"]?(?:\s|$|\.)",
        "format_json": r"(?:in|as|using)\s+json\s+format",
        "format_bullet": r"(?:in|as|using)\s+(?:bullet|bulleted)\s+(?:points?|list|format)",
        "all_caps": r"(?:in|using|write in)\s+(?:all\s+)?(?:caps|capitals|uppercase)",
        "no_caps": r"(?:in|using|write in)\s+(?:all\s+)?lowercase",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.strict = self.config.get("strict", True)

    def _extract_instructions(self, text: str) -> List[Dict[str, Any]]:
        """Extract verifiable instructions from input"""
        instructions = []
        text_lower = text.lower()

        for inst_type, pattern in self.INSTRUCTION_PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match:
                instructions.append({
                    "type": inst_type,
                    "value": match.group(1) if match.groups() else True,
                    "pattern": pattern,
                })

        return instructions

    def _verify_instruction(
        self,
        instruction: Dict[str, Any],
        output: str
    ) -> bool:
        """Verify if output follows instruction"""
        inst_type = instruction["type"]
        value = instruction["value"]
        output_lower = output.lower()

        if inst_type == "word_count":
            target = int(value)
            actual = len(output.split())
            tolerance = 0 if self.strict else int(target * 0.1)
            return abs(actual - target) <= tolerance

        elif inst_type == "sentence_count":
            target = int(value)
            # Count sentences by periods, exclamation marks, question marks
            sentences = re.split(r'[.!?]+', output)
            actual = len([s for s in sentences if s.strip()])
            tolerance = 0 if self.strict else 1
            return abs(actual - target) <= tolerance

        elif inst_type == "no_word":
            forbidden = value.lower()
            return forbidden not in output_lower

        elif inst_type == "must_include":
            required = value.lower()
            return required in output_lower

        elif inst_type == "start_with":
            required = value.lower()
            return output_lower.strip().startswith(required)

        elif inst_type == "end_with":
            required = value.lower()
            # Remove trailing punctuation for comparison
            output_cleaned = re.sub(r'[.!?]+$', '', output_lower.strip())
            return output_cleaned.endswith(required)

        elif inst_type == "format_json":
            try:
                import json
                # Try to find JSON in output
                json_match = re.search(r'\{.*\}|\[.*\]', output, re.DOTALL)
                if json_match:
                    json.loads(json_match.group())
                    return True
                return False
            except:
                return False

        elif inst_type == "format_bullet":
            # Check for bullet points
            bullet_patterns = [r'^\s*[-*•]\s', r'^\s*\d+[.)]\s']
            lines = output.split('\n')
            bullet_count = sum(1 for line in lines if any(re.match(p, line) for p in bullet_patterns))
            return bullet_count >= 2

        elif inst_type == "all_caps":
            # Check if mostly uppercase
            letters = [c for c in output if c.isalpha()]
            if not letters:
                return True
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            return upper_ratio > 0.9

        elif inst_type == "no_caps":
            # Check if all lowercase
            letters = [c for c in output if c.isalpha()]
            if not letters:
                return True
            lower_ratio = sum(1 for c in letters if c.islower()) / len(letters)
            return lower_ratio > 0.9

        return True  # Unknown instruction type

    def _compute_single(self, sample: EvalSample) -> Dict[str, Any]:
        """Compute IFEval for single sample"""
        instructions = self._extract_instructions(sample.input)

        if not instructions:
            return {"score": 1.0, "num_instructions": 0, "followed": 0}

        followed = 0
        results = []

        for inst in instructions:
            is_followed = self._verify_instruction(inst, sample.output)
            if is_followed:
                followed += 1
            results.append({
                "type": inst["type"],
                "followed": is_followed,
            })

        score = followed / len(instructions)

        return {
            "score": score,
            "num_instructions": len(instructions),
            "followed": followed,
            "details": results,
        }

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute IFEval for all samples"""
        if not samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples provided"},
            )

        per_sample_scores = []
        total_instructions = 0
        total_followed = 0

        for sample in samples:
            result = self._compute_single(sample)
            per_sample_scores.append(result["score"])
            total_instructions += result["num_instructions"]
            total_followed += result["followed"]

        avg_score = sum(per_sample_scores) / len(per_sample_scores)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={
                "total_instructions": total_instructions,
                "total_followed": total_followed,
                "instruction_follow_rate": total_followed / total_instructions if total_instructions > 0 else 1.0,
                "num_samples": len(samples),
            },
            per_sample_scores=per_sample_scores,
        )
