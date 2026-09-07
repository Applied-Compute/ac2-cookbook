import os

from ac2.sdk import Client, DatasetNotFoundError

from wordle.dataset import PROJECT, build_datasets


def main() -> None:
    if os.environ.get("AC2_MODE") != "prod":
        raise SystemExit("Set AC2_MODE=prod before uploading datasets.")
    client = Client(project=PROJECT)
    for name, tasks in build_datasets().items():
        try:
            client.datasets.get(name)
        except DatasetNotFoundError:
            client.datasets.create(name)
        client.datasets.add_tasks(name, tasks=tasks)
        print(f"{name}: {len(client.datasets.active_tasks(name))} tasks")


if __name__ == "__main__":
    main()
