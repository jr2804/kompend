"""Plan-cache keys and the off-path summary-deps golden values (knox-u84v.3).

The ``summarizer_plan`` layer keys one traversal plan per tree: the deps
combine the tree identity, the task identity, the model, and the policy, so
a plan from another task statement — or another tree, model, or budget —
can never match. The off-path summary deps (``NodeSummarizer._cache_deps``)
are pinned byte-for-byte so the traversal work cannot silently invalidate
existing vault caches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from kompend.chunking.summarizer import NodeSummarizer
from kompend.core.cache.store import (
    SUMMARIZER_PLAN_LAYER,
    make_cache_key,
    summarizer_plan_deps,
    summarizer_policy_key,
    summarizer_task_hash,
    summarizer_tree_hash,
)


def _plan_key(*, task: str = "Summarize the API", model: str = "m", include_model: bool = True) -> str:
    tree_hash = summarizer_tree_hash("root", [("a", ["Doc", "A"]), ("b", ["Doc", "B"])])
    task_hash = summarizer_task_hash(task, "title: API")
    deps = summarizer_plan_deps(
        tree_hash=tree_hash,
        task_hash=task_hash,
        model=model,
        policy=summarizer_policy_key("auto", max_nodes=200, max_windows=2000),
        include_model=include_model,
    )
    return make_cache_key(deps)


class TestSummarizerPlanLayer:
    def test_layer_name(self) -> None:
        assert SUMMARIZER_PLAN_LAYER == "summarizer_plan"

    def test_stable_across_reruns(self) -> None:
        assert _plan_key() == _plan_key()
        assert _plan_key() == "b3bf7835c6182183"

    def test_differs_across_task_statements(self) -> None:
        assert _plan_key(task="Summarize the API") != _plan_key(task="Summarize the CLI")

    def test_differs_across_trees(self) -> None:
        left = summarizer_tree_hash("root", [("a", ["Doc", "A"])])
        right = summarizer_tree_hash("root", [("a", ["Doc", "B"])])

        assert left != right

    def test_differs_across_models(self) -> None:
        assert _plan_key(model="m1") != _plan_key(model="m2")

    def test_differs_across_policies(self) -> None:
        narrow = summarizer_policy_key("auto", max_nodes=10, max_windows=2000)
        wide = summarizer_policy_key("auto", max_nodes=200, max_windows=2000)

        assert narrow != wide

    def test_cache_ignore_model_drops_the_model(self) -> None:
        """``compile.cache_ignore_model`` keeps its meaning for the plan layer."""
        without_model = _plan_key(model="m1", include_model=False)

        assert without_model == _plan_key(model="m2", include_model=False)
        assert without_model != _plan_key(model="m1", include_model=True)

    def test_task_hash_covers_the_frontmatter(self) -> None:
        assert summarizer_task_hash("t", "title: A") != summarizer_task_hash("t", "title: B")


class TestOffPathSummaryDepsGolden:
    """The traversal work must not move the off-path summary deps."""

    @staticmethod
    def _deps(content: str, heading: str, model: str, ignore_model: bool) -> dict[str, str]:
        agent = SimpleNamespace(model_string=model)
        summarizer = cast("Any", SimpleNamespace(_agent=agent, _cache_ignore_model=ignore_model))
        return NodeSummarizer._cache_deps(summarizer, content, heading)

    def test_off_path_deps_are_byte_identical(self) -> None:
        deps = self._deps("Hello world", "Doc / Section", "my-model", False)

        assert deps == {
            "content_hash": "64ec88ca00b268e5",
            "heading_chain": "70d77323113d8158",
            "model": "my-model",
        }

    def test_off_path_deps_without_model(self) -> None:
        deps = self._deps("Hello world", "Doc / Section", "my-model", True)

        assert deps == {
            "content_hash": "64ec88ca00b268e5",
            "heading_chain": "70d77323113d8158",
        }
