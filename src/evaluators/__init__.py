"""
HumanEvalComm V2 Evaluators Framework

A comprehensive framework for evaluating AI-generated code quality through
multiple complementary approaches including multi-LLM judges, static analysis,
dynamic testing, and sandboxed execution.
"""

__version__ = "2.0.0"
__author__ = "Jie JW Wu, Fatemeh H. Fard"
__description__ = ("Benchmarking the Communication Competence of Code "
                   "Generation for LLMs and LLM Agent")

from .aggregator import Aggregator
from .automated_static_dynamic import AutomatedStaticDynamic
from .calibration import Calibration
from .multi_llm_judge import MultiLLMJudge
from .sandbox_runner import SandboxRunner

__all__ = [
    "MultiLLMJudge",
    "AutomatedStaticDynamic",
    "SandboxRunner",
    "Calibration",
    "Aggregator",
]
