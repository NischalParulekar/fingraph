"""Evidence verifier.

Standalone: no LangGraph imports, so the false-completion benchmark can reuse it.
"""

from fingraph.verifier.core import Calculation, Claim, Evidence, Report, verify

__all__ = ["Calculation", "Claim", "Evidence", "Report", "verify"]
