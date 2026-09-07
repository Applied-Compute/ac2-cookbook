from ac2.runtime import EnvironmentProtocol, Grader, GraderOutput, Trace

from .environment import WordleEnvironment


class WordleGrader(Grader):
    async def _grade(
        self, grader_params: dict | None, trace: Trace, env: EnvironmentProtocol,
    ) -> GraderOutput:
        if not isinstance(env, WordleEnvironment):
            raise TypeError("WordleGrader requires WordleEnvironment.")
        # Integer numerator avoids 1.1 - 0.1*k floating-point endpoint drift.
        score = (11 - env.guess_count) / 10 if env.solved else 0.0
        return GraderOutput(
            score=score,
            reasoning=f"{env.terminated_reason or 'unfinished'} after {env.guess_count} guesses",
            artifacts={"solved": env.solved, "guesses": env.guess_count,
                       "terminated_reason": env.terminated_reason},
        )
