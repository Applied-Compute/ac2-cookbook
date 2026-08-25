from __future__ import annotations

from ac2.runtime import Environment, Grader, GraderOutput, Trace

from .environment import RPSEnvironment


class WinrateGrader(Grader):
    async def _grade(
        self,
        grader_params: dict | None,
        trace: Trace,
        env: Environment,
    ) -> GraderOutput:
        assert isinstance(env, RPSEnvironment), f"WinrateGrader expected RPSEnvironment, got {type(env).__name__}"
        total = env.wins + env.losses + env.ties
        if total == 0:
            return GraderOutput(score=0.0, reasoning="No rounds played.")
        winrate = env.wins / total
        return GraderOutput(
            score=winrate,
            reasoning=f"{env.wins}W/{env.losses}L/{env.ties}T over {total} rounds. Winrate: {winrate:.2%}",
        )
