"""Where the `ladders` workspace is, from inside the repo it symlinks to.

This is not a kernel concern and it does not live under ``kernels/`` for that reason: every
instrument in this repo that has to reach ``infra/``, ``experiments/`` or ``oss/`` needs it, and
``tests/kernels/test_kernel_dispatch_boundary.py`` correctly refuses to let production code reach
into a private kernel backend to borrow a path helper.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/scratch_llm/_workspace.py -> src/scratch_llm -> src -> repo root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    """The `ladders` workspace directory — the one holding infra/ and experiments/.

    It cannot be derived by walking up from this file. ``scratch_llm`` is a SYMLINK inside the
    workspace pointing at a sibling directory, so ``Path(__file__).resolve()`` lands outside the
    workspace entirely and ``parents[6]`` is the symlink's target's parent, not the workspace.
    Getting this wrong is silent: the drydock tests look for artifacts under a path that does not
    exist, skip, and report green forever — a test that cannot fail.

    So: try the candidates, and accept the first that actually contains the workspace's own files.
    """
    marker = Path("infra") / "drydock.sh"
    env = os.environ.get("LADDERS_ROOT")
    if env and (Path(env) / marker).is_file():
        return Path(env)

    # The workspace is not an ANCESTOR of this repo, it is a sibling that links to it — and
    # os.getcwd() hands back the resolved path, so walking up from cwd never finds it either.
    # What does identify it uniquely is the link itself: the workspace is the directory whose
    # `scratch_llm` entry resolves to this repo.
    for parent in (_REPO_ROOT.parent, *_REPO_ROOT.parents):
        for child in sorted(parent.iterdir()) if parent.is_dir() else []:
            if not child.is_dir() or not (child / marker).is_file():
                continue
            link = child / _REPO_ROOT.name
            if link.exists() and link.resolve() == _REPO_ROOT:
                return child
        # Only one level up; beyond that we would be scanning the whole home directory.
        break

    # Last resort: an ancestor of cwd that holds the marker (true when the repo is not symlinked).
    for c in (Path.cwd(), *Path.cwd().parents):
        if (c / marker).is_file():
            return c
    raise RuntimeError(
        "cannot locate the ladders workspace (no directory with infra/drydock.sh links to "
        f"{_REPO_ROOT}). Set LADDERS_ROOT=/path/to/ladders."
    )
