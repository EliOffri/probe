import asyncio
import uuid
from rich.console import Console
from models.finding import AttackCategory, Finding, Severity, AttackResult
from engine.target import TargetAdapter
from engine.attacker import AttackerLLM, ALL_CATEGORIES_TO_RUN
from engine.judge import JudgeLLM

console = Console()

SEVERITY_COLORS = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "blue",
    Severity.NONE: "dim",
}


class AdaptiveLoop:
    """
    Runs multiple rounds of attacks per category.
    After each round, the attacker sees what worked and adapts.
    Dead ends get abandoned. Cracks get probed deeper.
    """

    def __init__(
        self,
        target: TargetAdapter,
        attacker: AttackerLLM,
        judge: JudgeLLM,
        rounds: int = 3,
        attacks_per_round: int = 10,
    ):
        self.target = target
        self.attacker = attacker
        self.judge = judge
        self.rounds = rounds
        self.attacks_per_round = attacks_per_round
        self.all_results: list[AttackResult] = []

    async def run_all(self, system_prompt: str) -> list[Finding]:
        """Run the full adaptive loop across all categories."""
        all_findings: list[Finding] = []

        for category in ALL_CATEGORIES_TO_RUN:
            findings = await self.run_category(system_prompt, category)
            all_findings.extend(findings)

        return all_findings

    async def run_category(
        self,
        system_prompt: str,
        category: AttackCategory,
    ) -> list[Finding]:
        """
        Run multiple rounds for a single category.
        Each round the attacker learns from the previous round's results.
        """
        console.print(
            f"\n[dim]Category:[/dim] [bold]{category.value if hasattr(category, 'value') else category}[/bold] "
            f"[dim]({self.rounds} rounds × {self.attacks_per_round} attacks)[/dim]"
        )

        findings: list[Finding] = []
        seen_prompts: set[str] = set()

        for round_num in range(1, self.rounds + 1):
            console.print(f"  [dim]Round {round_num}/{self.rounds}[/dim]")

            prompts = await self.attacker.generate(
                system_prompt=system_prompt,
                category=category,
                n=self.attacks_per_round,
            )

            # Skip prompts we already tried
            fresh_prompts = [p for p in prompts if p not in seen_prompts]
            seen_prompts.update(fresh_prompts)

            if not fresh_prompts:
                console.print("  [dim]No new prompts generated — moving on[/dim]")
                break

            results = await self._fire_batch(
                system_prompt=system_prompt,
                prompts=fresh_prompts,
                category=category,
                round_num=round_num,
            )

            round_violations = 0
            for result in results:
                self.all_results.append(result)
                self.attacker.record_result(
                    category=category,
                    prompt=result.prompt,
                    violated=result.violated,
                    severity=result.severity.value,
                )

                if result.violated and result.severity != Severity.NONE:
                    round_violations += 1
                    finding = self._to_finding(result)
                    if not self._is_duplicate(finding, findings):
                        findings.append(finding)
                        color = SEVERITY_COLORS[result.severity]
                        console.print(
                            f"    [{color}]{result.severity.value.upper()}[/{color}] "
                            f"{result.reasoning[:70]}"
                        )

            if round_violations == 0 and round_num > 1:
                console.print(
                    "  [dim]No new violations this round — stopping early[/dim]"
                )
                break

        return findings

    async def _fire_batch(
        self,
        system_prompt: str,
        prompts: list[str],
        category: AttackCategory,
        round_num: int,
    ) -> list[AttackResult]:
        """Fire all prompts in a batch concurrently."""

        async def attack_one(prompt: str, idx: int) -> AttackResult | None:
            attack_id = f"{category.value if hasattr(category, 'value') else category}_r{round_num}_{idx}_{uuid.uuid4().hex[:6]}"
            try:
                response = await self.target.send(prompt)
                console.print(f"\n[dim]PROMPT:[/dim] {prompt[:80]}")
                console.print(f"[dim]RESPONSE:[/dim] {response[:120]}\n")
                result = await self.judge.evaluate(
                    system_prompt=system_prompt,
                    attack_prompt=prompt,
                    response=response,
                    category=category,
                    attack_id=attack_id,
                    turn_number=round_num,
                )
                return result
            except Exception as e:
                console.print(f"    [dim red]Attack failed: {e}[/dim red]")
                return None

        results = await asyncio.gather(
            *[attack_one(p, i) for i, p in enumerate(prompts)]
        )
        return [r for r in results if r is not None]

    def _to_finding(self, result: AttackResult) -> Finding:
        return Finding(
            finding_id=result.attack_id,
            category=result.category,
            severity=result.severity,
            title=f"{result.severity.value.upper()}: "
                  f"{(result.category.value if hasattr(result.category, 'value') else result.category).replace('_', ' ').title()}",
            description=result.reasoning,
            prompt=result.prompt,
            response=result.response,
            evidence=result.evidence,
            tool_calls=result.tool_calls,
        )

    def _is_duplicate(self, new: Finding, existing: list[Finding]) -> bool:
        """Deduplicate findings with same category and very similar description."""
        for f in existing:
            if f.category == new.category and f.severity == new.severity:
                if new.description[:60] == f.description[:60]:
                    return True
        return False
