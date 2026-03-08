"""
Prompt Templates for Ground Truth Generation
"""
from typing import Dict, Any, Optional


class GTPromptTemplates:
    """Templates for generating ground truth using LLM"""

    # General QA ground truth
    QA_TEMPLATE = """You are an expert evaluator. Given the following question, provide the most accurate and concise answer.

Question: {input}

{context_section}

Provide only the answer, without any explanation or additional text."""

    # Summarization ground truth
    SUMMARIZATION_TEMPLATE = """You are an expert summarizer. Summarize the following text concisely while retaining all key information.

Text to summarize:
{input}

Provide a clear, accurate summary."""

    # Translation ground truth
    TRANSLATION_TEMPLATE = """You are a professional translator. Translate the following text to {target_language}.

Source text:
{input}

Provide only the translation, without any explanation."""

    # Tool calling ground truth
    TOOL_CALLING_TEMPLATE = """You are an AI assistant that can use tools. Given the following request and available tools, determine the correct tool call(s).

User request: {input}

Available tools:
{tools}

Respond with the exact tool call in JSON format:
{{"tool": "tool_name", "arguments": {{...}}}}

Only respond with the JSON, no explanation."""

    # Code generation ground truth
    CODE_TEMPLATE = """You are an expert programmer. Write clean, correct code for the following task.

Task: {input}

{language_section}

Provide only the code, without explanation."""

    # Multiple choice answer
    MULTIPLE_CHOICE_TEMPLATE = """You are an expert in {domain}. Answer the following multiple choice question.

Question: {input}

Choices:
{choices}

Respond with only the letter of the correct answer (A, B, C, etc.)."""

    # RAG answer generation
    RAG_TEMPLATE = """Based on the provided context, answer the following question accurately.

Context:
{context}

Question: {input}

Important: Only use information from the context. If the answer is not in the context, say "Cannot be determined from the context."

Answer:"""

    @classmethod
    def get_qa_prompt(cls, input_text: str, context: Optional[str] = None) -> str:
        context_section = f"Context:\n{context}" if context else ""
        return cls.QA_TEMPLATE.format(input=input_text, context_section=context_section)

    @classmethod
    def get_summarization_prompt(cls, input_text: str) -> str:
        return cls.SUMMARIZATION_TEMPLATE.format(input=input_text)

    @classmethod
    def get_translation_prompt(cls, input_text: str, target_language: str = "English") -> str:
        return cls.TRANSLATION_TEMPLATE.format(input=input_text, target_language=target_language)

    @classmethod
    def get_tool_calling_prompt(cls, input_text: str, tools: str) -> str:
        return cls.TOOL_CALLING_TEMPLATE.format(input=input_text, tools=tools)

    @classmethod
    def get_code_prompt(cls, input_text: str, language: Optional[str] = None) -> str:
        language_section = f"Language: {language}" if language else ""
        return cls.CODE_TEMPLATE.format(input=input_text, language_section=language_section)

    @classmethod
    def get_multiple_choice_prompt(
        cls,
        input_text: str,
        choices: list,
        domain: str = "general knowledge"
    ) -> str:
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        return cls.MULTIPLE_CHOICE_TEMPLATE.format(
            input=input_text,
            choices=choices_text,
            domain=domain
        )

    @classmethod
    def get_rag_prompt(cls, input_text: str, context: str) -> str:
        return cls.RAG_TEMPLATE.format(input=input_text, context=context)

    @classmethod
    def get_prompt_for_task(
        cls,
        task_type: str,
        input_text: str,
        **kwargs
    ) -> str:
        """Get appropriate prompt template for task type"""
        if task_type == "qa":
            return cls.get_qa_prompt(input_text, kwargs.get("context"))
        elif task_type == "summarization":
            return cls.get_summarization_prompt(input_text)
        elif task_type == "translation":
            return cls.get_translation_prompt(input_text, kwargs.get("target_language", "English"))
        elif task_type == "tool_calling":
            return cls.get_tool_calling_prompt(input_text, kwargs.get("tools", ""))
        elif task_type == "coding":
            return cls.get_code_prompt(input_text, kwargs.get("language"))
        elif task_type == "reasoning":
            return cls.get_multiple_choice_prompt(
                input_text,
                kwargs.get("choices", []),
                kwargs.get("domain", "general knowledge")
            )
        elif task_type == "rag":
            return cls.get_rag_prompt(input_text, kwargs.get("context", ""))
        else:
            # Default: general QA
            return cls.get_qa_prompt(input_text, kwargs.get("context"))
