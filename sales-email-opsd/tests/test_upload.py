from unittest.mock import Mock

import pytest
from ac2.runtime import Message, Task
from ac2.sdk import DatasetNotFoundError

from upload_dataset import upload_bundle


def task(text: str) -> Task:
    return Task(input=[Message(role="user", content=text)])


def test_existing_snapshot_upload_is_a_noop() -> None:
    client = Mock()
    tasks = [task("First"), task("Second")]
    client.datasets.load_tasks.return_value = tasks
    upload_bundle(client, {"example-v2": tasks})
    client.datasets.create.assert_not_called()
    client.datasets.add_tasks.assert_not_called()


def test_interrupted_upload_resumes_only_missing_tasks() -> None:
    client = Mock()
    tasks = [task("First"), task("Second")]
    client.datasets.load_tasks.return_value = tasks[:1]
    upload_bundle(client, {"example-v2": tasks})
    client.datasets.add_tasks.assert_called_once_with("example-v2", tasks=tasks[1:])


def test_conflicting_snapshot_fails_before_any_writes() -> None:
    client = Mock()
    client.datasets.get.side_effect = [DatasetNotFoundError("new-v2"), Mock()]
    client.datasets.load_tasks.return_value = [task("Unexpected")]
    with pytest.raises(ValueError, match="different snapshot"):
        upload_bundle(client, {"new-v2": [task("New")], "existing-v2": [task("Expected")]})
    client.datasets.create.assert_not_called()
    client.datasets.add_tasks.assert_not_called()
