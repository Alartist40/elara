"""
Audit logging for the Constitutional Layer.
Records all filtering decisions for transparency and accountability.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


class AuditLogger:
    """
    Immutable audit trail for constitutional filtering decisions.
    All decisions are logged with timestamps, principle IDs, and actions taken.
    """

    def __init__(self, log_path: str = "logs/constitution.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger("elara.constitutional.audit")
        self.logger.setLevel(logging.INFO)

        # File handler (append mode, never overwrite)
        if not self.logger.handlers:
            handler = logging.FileHandler(
                str(self.log_path), mode="a", encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._entry_count = 0

    def log_decision(
        self,
        stage: str,          # "pre_filter" or "post_filter"
        input_text: str,
        decision: str,       # "allow", "block", "flag", "rewrite"
        principle_id: Optional[str] = None,
        principle_category: Optional[str] = None,
        action_taken: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a filtering decision."""
        self._entry_count += 1

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "entry_id": self._entry_count,
            "stage": stage,
            "input_preview": input_text[:200] + "..." if len(input_text) > 200 else input_text,
            "decision": decision,
            "principle_id": principle_id,
            "principle_category": principle_category,
            "action_taken": action_taken,
            "metadata": metadata or {},
        }

        self.logger.info(json.dumps(entry, ensure_ascii=False))

    def log_violation(
        self,
        principle_id: str,
        input_text: str,
        severity: str = "medium",
        details: Optional[str] = None,
    ) -> None:
        """Log a principle violation (more prominent than a decision)."""
        self.log_decision(
            stage="violation",
            input_text=input_text,
            decision="blocked",
            principle_id=principle_id,
            metadata={
                "severity": severity,
                "details": details or "Principle violation detected.",
            },
        )

    def get_session_stats(self) -> Dict[str, int]:
        """Return statistics for current session."""
        return {
            "session_id": self._session_id,
            "total_entries": self._entry_count,
        }
