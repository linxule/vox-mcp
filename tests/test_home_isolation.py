"""
The test suite must never write into the user's real ~/.vox.

This is the check that would have caught it. The suite had been appending JSONL
into the user's actual conversation store for as long as anyone had run it — 151
test-fixture threads (all stamped with the hardcoded 2023-01-01 `created_at`) sat
in a directory that also held 976 real conversations. Nothing failed, because
polluting a directory is silent.

It surfaced only as a bizarre CI failure: test_add_turn_success asserted that
storage.get() was called once, which is true ONLY when the thread's JSONL already
exists. On a clean runner it does not, so the persistence layer takes its
retroactive-header path and reads storage a second time — and the test fails. That
same run then writes the file, so every subsequent run on that machine passes. A
test that fails once on a fresh checkout and is green forever after is not a test.

So the isolation gets a gate of its own, rather than living as an autouse fixture
that any future edit could quietly drop.
"""

from pathlib import Path

import config
from utils.thread_persistence import _get_threads_dir

# The location the production default resolves to. Nothing under here may be
# touched by a test run.
REAL_VOX_HOME = (Path.home() / ".vox").resolve()


def _is_inside_real_vox(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == REAL_VOX_HOME or REAL_VOX_HOME in resolved.parents


def test_threads_dir_is_redirected_away_from_the_users_real_vox_home():
    """
    The persistence seam. If the autouse isolate_vox_threads_dir fixture is
    removed or stops working, this resolves to ~/.vox/threads and fails.
    """
    threads_dir = _get_threads_dir()
    assert not _is_inside_real_vox(threads_dir), (
        f"The test suite is writing into the user's real thread store: {threads_dir}\n"
        "conftest.py's autouse `isolate_vox_threads_dir` fixture must redirect "
        "config.VOX_THREADS_DIR to a tmp_path."
    )


def test_markdown_export_dir_is_also_redirected():
    """
    The second consumer. utils.markdown_export builds its output path from
    config.VOX_THREADS_DIR independently of _get_threads_dir(), so patching only
    the persistence helper would leave this one still writing to the real home
    directory. Both must be covered by the same seam.
    """
    export_dir = config.VOX_THREADS_DIR / "exports"
    assert not _is_inside_real_vox(export_dir), (
        f"Markdown exports would land in the user's real ~/.vox tree: {export_dir}"
    )


def test_the_guard_is_not_vacuous():
    """
    Coverage before verdict: prove _is_inside_real_vox actually detects the thing
    it is supposed to detect. A guard that returns False for every input would let
    both tests above pass while the suite happily polluted the home directory.
    """
    assert _is_inside_real_vox(Path.home() / ".vox" / "threads")
    assert _is_inside_real_vox(Path.home() / ".vox")
    assert not _is_inside_real_vox(Path("/tmp/somewhere-else"))
