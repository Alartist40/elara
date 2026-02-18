"""
ConstitutionalLayer - Immutable safety and value alignment.
Pre-filters input and post-filters output based on biblical principles.
Implemented as code, not model weights (auditable, version-controlled).
"""

from typing import Tuple, Dict, Any, Optional, List
from elara_core.constitutional.principles import Principle, PrincipleLoader
from elara_core.constitutional.audit import AuditLogger


class FilterResult:
    """Result of a constitutional filtering operation."""

    def __init__(
        self,
        allowed: bool,
        text: str,
        triggered_principles: List[str],
        action_taken: str = "none",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.allowed = allowed
        self.text = text
        self.triggered_principles = triggered_principles
        self.action_taken = action_taken
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "triggered_principles": self.triggered_principles,
            "action_taken": self.action_taken,
            "metadata": self.metadata,
        }


class ConstitutionalLayer:
    """
    Immutable constitutional safety layer.
    Enforces biblical principles as code-based rules.

    This layer is:
    - Immutable: Rules cannot be changed by model weights or user input.
    - Auditable: Every decision is logged with full context.
    - Rule-based: No ML, pure pattern matching and logic.
    """

    def __init__(
        self,
        principles_path: str,
        audit_log_path: str = "logs/constitution.log",
        strict_mode: bool = True,
        watermark_voice: bool = True,
    ):
        self.loader = PrincipleLoader(principles_path)
        self.audit = AuditLogger(audit_log_path)
        self.strict_mode = strict_mode
        self.watermark_voice = watermark_voice

    def pre_filter(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> FilterResult:
        """
        Pre-filter input before it reaches the model.

        Args:
            text: Input text to filter.
            context: Additional context (modality, voice_input, etc.)

        Returns:
            FilterResult with decision and metadata.
        """
        context = context or {}
        ctx_type = context.get("modality", "text")
        triggered = []

        # Check all applicable principles
        applicable = self.loader.get_by_context("all")
        if ctx_type:
            applicable += self.loader.get_by_context(str(ctx_type).lower())

        # Deduplicate by principle ID
        seen_ids = set()
        unique_applicable = []
        for p in applicable:
            if p.id not in seen_ids:
                seen_ids.add(p.id)
                unique_applicable.append(p)

        for principle in unique_applicable:
            if principle.matches(text):
                triggered.append(principle)

        if not triggered:
            # No violations — allow
            self.audit.log_decision(
                stage="pre_filter",
                input_text=text,
                decision="allow",
            )
            return FilterResult(
                allowed=True,
                text=text,
                triggered_principles=[],
                action_taken="none",
            )

        # Process triggered principles
        blocked = False
        flagged = False
        modified_text = text
        triggered_ids = []
        actions = []

        for principle in triggered:
            triggered_ids.append(principle.id)

            if principle.rule_type == "block":
                blocked = True
                actions.append(f"block:{principle.id}")
                self.audit.log_violation(
                    principle_id=principle.id,
                    input_text=text,
                    severity="high",
                    details=f"Blocked by principle: {principle.scriptural_basis}",
                )

            elif principle.rule_type == "flag":
                flagged = True
                actions.append(f"flag:{principle.id}")
                self.audit.log_decision(
                    stage="pre_filter",
                    input_text=text,
                    decision="flag",
                    principle_id=principle.id,
                    principle_category=principle.category,
                    action_taken=principle.action,
                )

        if blocked:
            # Use the first blocking principle's response
            block_response = next(
                p.response for p in triggered if p.rule_type == "block"
            )
            return FilterResult(
                allowed=False,
                text=block_response,
                triggered_principles=triggered_ids,
                action_taken="block",
                metadata={"original_input_preview": text[:100]},
            )

        # Flagged but not blocked
        return FilterResult(
            allowed=True,
            text=modified_text,
            triggered_principles=triggered_ids,
            action_taken="flag",
            metadata={"flags": actions},
        )

    def post_filter(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> FilterResult:
        """
        Post-filter model output before it reaches the user.

        Args:
            text: Model output text to filter.
            context: Additional context (tier, modality, etc.)

        Returns:
            FilterResult with decision, potentially modified text, and metadata.
        """
        context = context or {}
        triggered = []
        modified_text = text
        triggered_ids = []
        actions = []

        # Check all applicable principles against output
        for principle in self.loader.principles:
            if principle.matches(text):
                triggered.append(principle)

        for principle in triggered:
            triggered_ids.append(principle.id)

            if principle.rule_type == "block":
                # Re-block harmful output
                self.audit.log_violation(
                    principle_id=principle.id,
                    input_text=text,
                    severity="high",
                    details="Output contained blocked content.",
                )
                return FilterResult(
                    allowed=False,
                    text="[Response removed by safety policy]",
                    triggered_principles=triggered_ids,
                    action_taken="block_output",
                )

            elif principle.action == "add_disclaimer":
                modified_text = (
                    modified_text + f"\n\n⚠️ Disclaimer: {principle.response}"
                )
                actions.append(f"disclaimer:{principle.id}")

        # Voice watermark
        if self.watermark_voice and context.get("voice_output"):
            actions.append("voice_watermark")

        if not triggered:
            self.audit.log_decision(
                stage="post_filter",
                input_text=text,
                decision="allow",
            )
        else:
            self.audit.log_decision(
                stage="post_filter",
                input_text=text,
                decision="flag" if actions else "allow",
                metadata={"actions": actions},
            )

        return FilterResult(
            allowed=True,
            text=modified_text,
            triggered_principles=triggered_ids,
            action_taken="; ".join(actions) if actions else "none",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return constitutional layer statistics."""
        return {
            "total_principles": len(self.loader),
            "categories": list(self.loader.categories.keys()),
            "strict_mode": self.strict_mode,
            "audit_stats": self.audit.get_session_stats(),
        }
