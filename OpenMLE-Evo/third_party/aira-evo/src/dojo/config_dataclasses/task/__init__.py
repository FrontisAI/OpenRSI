# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class _LazyTaskMap(Mapping[str, Any]):
    def _materialize(self) -> dict[str, Any]:
        from dojo.tasks.mlebench.task import MLEBenchTask

        return {"MLEBenchTaskConfig": MLEBenchTask}

    def __getitem__(self, key: str) -> Any:
        return self._materialize()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._materialize())

    def __len__(self) -> int:
        return len(self._materialize())


TASK_MAP: Mapping[str, Any] = _LazyTaskMap()
