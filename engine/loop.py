import asyncio
import uuid
from rich.console import Console
from models.finding import AttackCategory, Finding, Severity, AttackResult, TargetProfile
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
    Supports single-shot and multi-turn attack modes.
    """

    def __init__(
        self,
        target: TargetAdapter,
        attacker: AttackerLLM,
        judge: JudgeLLM,
        rounds: int = 3,
        attacks_per_round: int = 10,
        target_context: str = "",
        max_turns: int = 1,
        target_profile: TargetProfile = None,
    ):
        self.target = target
        self.attacker = attacker
        self.judge = judge
        self.rounds = rounds
        self.attacks_per_round = attacks_per_round
        self.target_context = target_context
        self.max_turns = max_turns
        self.target_profile = target_profile or TargetProfile()
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
        mode = f"multi-turn (max {self.max_turns})" if self.max_turns > 1 else "single-shot"
        console.print(
            f"\n[dim]Category:[/dim] [bold]{category.value}[/bold] "
            f"[dim]({self.rounds} rounds × {self.attacks_per_round} attacks, {mode})[/dim]"
        )

        findings: list[Finding] = []
        seen_prompts: set[str] = set()

        for round_num in range(1, self.rounds + 1):
            console.print(f"  [dim]Round {round_num}/{self.rounds}[/dim]")

            if self.max_turns > 1:
                results = await self._run_multi_turn_round(
                    system_prompt, category, round_num, seen_prompts
                )
            else:
                results = await self._run_single_turn_round(
                    system_prompt, category, round_num, seen_prompts
                )

            if results is None:
                break

            round_violations = 0
            for result in results:
                self.all_results.append(result)
                self.attacker.record_result(
                    category=category,
                    prompt=result.prompt,
                    violated=result.violated,
                    severity=result.severity.value,
                    target_profile=self.target_profile,
                )

                if result.violated and result.severity != Severity.NONE:
                    round_violations += 1
                    finding = self._to_finding(result)
                    if not self._is_duplicate(finding, findings):
                        findings.append(finding)
                        color = SEVERITY_COLORS[result.severity]
                        turn_label = f"[dim](turn {result.turn_number})[/dim] " if self.max_turns > 1 else ""
                        console.print(
                            f"    [{color}]{result.severity.value.upper()}[/{color}] "
                            f"{turn_label}{result.reasoning[:70]}"
                        )

            if round_violations == 0 and round_num > 1:
                console.print("  [dim]No new violations this round — stopping early[/dim]")
                break

        return findings

    async def _run_single_turn_round(
        self,
        system_prompt: str,
        category: AttackCategory,
        round_num: int,
        seen_prompts: set[str],
    ) -> list[AttackResult] | None:
        prompts = await self.attacker.generate(
            system_prompt=system_prompt,
            category=category,
            n=self.attacks_per_round,
            target_context=self.target_context,
            target_profile=self.target_profile,
        )

        fresh = [p for p in prompts if p not in seen_prompts]
        seen_prompts.update(fresh)

        if not fresh:
            console.print("  [dim]No new prompts generated — moving on[/dim]")
            return None

        return await self._fire_single_batch(
            system_prompt=system_prompt,
            prompts=fresh,
            category=category,
            round_num=round_num,
        )

    async def _run_multi_turn_round(
        self,
        system_prompt: str,
        category: AttackCategory,
        round_num: int,
        seen_prompts: set[str],
    ) -> list[AttackResult] | None:
        plans = await self.attacker.generate_conversation_plans(
            system_prompt=system_prompt,
            category=category,
            n=self.attacks_per_round,
            max_turns=self.max_turns,
            target_context=self.target_context,
            target_profile=self.target_profile,
        )

        fresh_plans = [p for p in plans if p and p[0] not in seen_prompts]
        for p in fresh_plans:
            seen_prompts.add(p[0])

        if not fresh_plans:
            console.print("  [dim]No new conversation plans generated — moving on[/dim]")
            return None

        return await self._fire_multi_turn_batch(
            system_prompt=system_prompt,
            plans=fresh_plans,
            category=category,
            round_num=round_num,
        )

    async def _fire_single_batch(
        self,
        system_prompt: str,
        prompts: list[str],
        category: AttackCategory,
        round_num: int,
    ) -> list[AttackResult]:
        async def attack_one(prompt: str, idx: int) -> AttackResult | None:
            attack_id = f"{category.value}_r{round_num}_{idx}_{uuid.uuid4().hex[:6]}"
            try:
                response = await self.target.send(prompt)
                return await self.judge.evaluate(
                    system_prompt=system_prompt,
                    attack_prompt=prompt,
                    response=response,
                    category=category,
                    attack_id=attack_id,
                    turn_number=round_num,
                )
            except Exception as e:
                console.print(f"    [dim red]Attack failed: {e}[/dim red]")
                return None

        results = await asyncio.gather(*[attack_one(p, i) for i, p in enumerate(prompts)])
        return [r for r in results if r is not None]

    async def _fire_multi_turn_batch(
        self,
        system_prompt: str,
        plans: list[list[str]],
        category: AttackCategory,
        round_num: int,
    ) -> list[AttackResult]:
        async def run_plan(plan: list[str], idx: int) -> AttackResult | None:
            attack_id = f"{category.value}_r{round_num}_{idx}_{uuid.uuid4().hex[:6]}"
            try:
                conversation = await self._execute_conversation(plan)
                console.print(
                    f"    [dim]Conversation {idx+1}: {len(conversation)//2} turns[/dim]"
                )
                return await self.judge.evaluate_conversation(
                    system_prompt=system_prompt,
                    conversation=conversation,
                    category=category,
                    attack_id=attack_id,
                )
            except Exception as e:
                console.print(f"    [dim red]Conversation failed: {e}[/dim red]")
                return None

        results = await asyncio.gather(*[run_plan(p, i) for i, p in enumerate(plans)])
        return [r for r in results if r is not None]

    async def _execute_conversation(self, turns: list[str]) -> list[dict]:
        """Execute a conversation plan against the target, returning the full history."""
        history = []
        for turn_msg in turns:
            response = await self.target.send(turn_msg, history=history)
            history.append({"role": "user", "content": turn_msg})
            history.append({"role": "assistant", "content": response})
        return history

    def _to_finding(self, result: AttackResult) -> Finding:
        return Finding(
            finding_id=result.attack_id,
            category=result.category,
            severity=result.severity,
            title=f"{result.severity.value.upper()}: "
                  f"{result.category.value.replace('_', ' ').title()}",
            description=result.reasoning,
            prompt=result.prompt,
            response=result.response,
            evidence=result.evidence,
            tool_calls=result.tool_calls,
            remediation=result.remediation,
            conversation_turns=result.turn_number,
        )

    def _is_duplicate(self, new: Finding, existing: list[Finding]) -> bool:
        for f in existing:
            if f.category == new.category and f.severity == new.severity:
                if new.description[:60] == f.description[:60]:
                    return True
        return False
