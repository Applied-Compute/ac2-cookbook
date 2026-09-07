"""Train Qwen3.6-35B-A3B on four training + four inference B200 GPUs."""

import json
import shlex
from dataclasses import asdict, replace
from pathlib import Path

from .train import CONFIG as BASE_CONFIG, main

CONFIG = replace(
    BASE_CONFIG,
    model="Qwen/Qwen3.6-35B-A3B",
    ac2_agent="WordleQwen36Agent",
    training_agent_names=["WordleQwen36Agent"],
    n_training_replicas=1,
    n_inference_replicas=2,
    # One eval replica is a separate two-GPU inference engine.
    n_eval_replicas=1,
    name="wordle-qwen36-35b-a3b-modal-40steps-eval5",
    extra_train_args={
        "save_interval": 5,
        # Replica size = PP * max(TP * CP, EP * ETP) = 4.
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "rollout_num_gpus_per_engine": 2,
        "sglang_expert_parallel_size": 1,
        "optimizer_cpu_offload": True,
        "overlap_cpu_optimizer_d2h_h2d": True,
        "use_precision_aware_optimizer": True,
        # Fit one long game plus packing headroom; avoid packing 262k tokens
        # onto each actor just because the catalog supports longer contexts.
        "max_tokens_per_gpu": 33792,
        "micro_batch_size": 1,
        "recompute_num_layers": 1,
        "sglang_mem_fraction_static": 0.7,
        # At most 32 live games across two engines (also 32 in each eval).
        "sglang_cuda_graph_bs": [1, 2, 4, 8, 16, 32],
    },
)


def validate_allocation(resources, command: list[str]) -> dict:
    """Reject accidental replica multiplication before waiting for startup."""
    flags = {}
    for token in shlex.split(command[-1]):
        if token.startswith("--") and "=" in token:
            key, value = token[2:].split("=", 1)
            flags[key] = value
    expected = {
        "actor-num-nodes": "1", "actor-num-gpus-per-node": "4",
        "rollout-num-gpus": "4", "rollout-num-gpus-per-engine": "2",
        "tensor-model-parallel-size": "4", "context-parallel-size": "1",
        "expert-model-parallel-size": "4", "expert-tensor-parallel-size": "1",
    }
    if resources.num_nodes != 1 or int(resources.gpus) != 8 or resources.gpu_type != "b200":
        raise RuntimeError(f"Expected one eight-B200 node, received {resources}")
    for key, value in expected.items():
        if flags.get(key) != value:
            raise RuntimeError(f"Unexpected {key}: {flags.get(key)!r}; expected {value}")
    return expected


def verify_run(client, train_id: str, output: Path) -> None:
    run = client.train.get(train_id)
    job = client.jobs.get(run.job_id)
    spec = client.jobs.get_spec(run.job_id)
    topology = validate_allocation(job.spec.resources, spec.command)
    # Store the command and resource facts, never credentials from spec.env.
    (output / "allocation.json").write_text(json.dumps({
        "job_id": job.id, "resources": asdict(job.spec.resources),
        "topology": topology, "command": spec.command,
    }, indent=2) + "\n")
    print("Verified allocation: one node, 4 training + 4 inference B200 GPUs.", flush=True)


if __name__ == "__main__":
    main(CONFIG, verify_run=verify_run, default_max_minutes=235)
