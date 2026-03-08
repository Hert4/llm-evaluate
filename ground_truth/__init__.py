"""
Ground Truth Generation Module
==============================
Uses OpenAI-compatible API to generate ground truth for evaluation.
"""

from .generator import GroundTruthGenerator
from .prompts import GTPromptTemplates

__all__ = ["GroundTruthGenerator", "GTPromptTemplates"]
