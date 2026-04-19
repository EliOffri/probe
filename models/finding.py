from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class AttackCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    ROLE_CONFUSION = "role_confusion"
    DATA_EXTRACTION = "data_extraction"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    GOAL_HIJACKING = "goal_hijacking"
    TOOL_ABUSE = "tool_abuse"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SOCIAL_ENGINEERING = "social_engineering"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    CONTEXT_MANIPULATION = "context_manipulation"
    FUNCTIONAL_BYPASS = "functional_bypass"




class AttackResult(BaseModel):
    attack_id: str
    category: AttackCategory
    prompt: str
    response: str
    violated: bool
    severity: Severity
    reasoning: str
    evidence: Optional[str] = None
    turn_number: int = 1
    tool_calls: list[dict] = Field(default_factory=list)
    remediation: Optional[str] = None


class Finding(BaseModel):
    finding_id: str
    category: AttackCategory
    severity: Severity
    title: str
    description: str
    prompt: str
    response: str
    evidence: Optional[str] = None
    tool_calls: list[dict] = Field(default_factory=list)
    remediation: Optional[str] = None


class Run(BaseModel):
    run_id: str
    target_name: str
    target_system_prompt: str
    model: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_attacks: int = 0
    findings: list[Finding] = Field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def passed(self) -> bool:
        return self.critical_count == 0
