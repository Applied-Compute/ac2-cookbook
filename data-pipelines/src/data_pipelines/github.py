import os
from collections.abc import Sequence

import httpx
from ac2.runtime import Message, Task
from ac2.sdk.pipelines import PipelineSpec, pipeline, publish_dataset_output, step
from ac2.sdk.registry.datasets import Dataset
from openai import OpenAI
from pydantic import BaseModel, Field


class GeneratedTask(BaseModel):
    question: str
    reference_answer: str
    rubric: str


class Review(BaseModel):
    score: float = Field(ge=0, le=1)
    reasoning: str


def _openai() -> OpenAI:
    return OpenAI(timeout=60, max_retries=2)


@step
def merged_pull_requests(*, repository: str, limit: int = 3) -> list[dict[str, str]]:
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    url = f"https://api.github.com/repos/{repository}/pulls"
    response = httpx.get(
        url,
        params={"state": "closed", "per_page": limit, "sort": "updated", "direction": "desc"},
        headers={"Accept": "application/vnd.github+json"},
        timeout=20,
    )
    response.raise_for_status()
    pulls = response.json()
    return [
        {"url": pull["html_url"], "title": pull["title"], "body": pull["body"] or "", "patch_url": pull["patch_url"]}
        for pull in pulls
        if pull.get("merged_at")
    ]


@step
def make_tasks(*, pull_requests: Sequence[dict[str, str]]) -> list[Task]:
    model = _openai()
    tasks: list[Task] = []
    for pull_request in pull_requests:
        response = httpx.get(pull_request["patch_url"], timeout=20)
        response.raise_for_status()
        patch = response.text[:30_000]
        generated = model.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Create one self-contained code-review question from this public merged PR. Do not ask for unseen repository context. Give a concise reference answer and a rubric. Return only task material grounded in the diff."},
                {"role": "user", "content": f"PR: {pull_request['title']}\nDescription: {pull_request['body']}\nPatch:\n{patch}"},
            ],
            response_format=GeneratedTask,
        ).choices[0].message.parsed
        if generated is None:
            raise RuntimeError(f"Task generation returned no result for {pull_request['url']}")
        baseline = model.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": generated.question}],
        ).choices[0].message.content or ""
        review = model.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Score the answer against the rubric. Score 1 only when it meets the rubric; return a score from 0 to 1 and a short reason."},
                {"role": "user", "content": f"Rubric: {generated.rubric}\nReference: {generated.reference_answer}\nAnswer: {baseline}"},
            ],
            response_format=Review,
        ).choices[0].message.parsed
        if review is None:
            raise RuntimeError(f"Review returned no result for {pull_request['url']}")
        if review.score < 0.8:
            tasks.append(
                Task(
                    input=[Message(role="user", content=generated.question)],
                    grader_params={"reference_answer": generated.reference_answer, "rubric": generated.rubric},
                    tags={"source_pr": pull_request["url"], "baseline_score": review.score, "baseline_reason": review.reasoning},
                )
            )
    return tasks


class GitHubPRSyntheticDataset(BaseModel):
    dataset: Dataset


@pipeline(
    name="github-pr-synthetic-dataset",
    log_prints=True,
    ac2=PipelineSpec(
        secret_keys=("OPENAI_API_KEY",),
        output_dataset_name="github-pr-synthetic-failures",
        max_concurrent_runs=1,
        max_runtime_seconds=3600,
    ),
)
def github_pr_synthetic_dataset(*, repository: str = "<owner>/<public-repository>") -> GitHubPRSyntheticDataset:
    pulls = merged_pull_requests(repository=repository)
    tasks = make_tasks(pull_requests=pulls)
    print(f"Created {len(tasks)} tasks from {len(pulls)} merged pull requests")
    return GitHubPRSyntheticDataset(dataset=publish_dataset_output(tasks=tasks))
