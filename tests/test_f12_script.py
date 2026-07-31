"""F12 `scripts/f12_corpus_ablation.py` — hermetic CPU smoke of the operator entry point.

``--toy`` runs both arms end-to-end (stage → retrained BPE → tiny training → held-out bpb +
fixture CORE) with no network, writes each arm's JSON the moment it finishes, and lands the
final verdict JSON with the pre-registered falsifier/kill fields. The resume pass adopts the
first run's shard dirs via ``--fineweb-data-dir``/``--climbmix-data-dir`` and must still log
the overlap rate (recovered from build_stats.json).
"""

import json
import math
import sys
from pathlib import Path


def _f12():
    """Import the CLI module (scripts/ is not a package)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import f12_corpus_ablation
    finally:
        sys.path.pop(0)
    return f12_corpus_ablation


_TOY_ARGS = ["--toy", "--depth", "1", "--steps", "2", "--eval-every", "1", "--core"]


def test_toy_mode_end_to_end_verdict_json(tmp_path: Path) -> None:
    mod = _f12()
    out = tmp_path / "f12"
    mod.main([*_TOY_ARGS, "--out-dir", str(out)])

    verdict_path = out / "f12_verdict.json"
    payload = json.loads(verdict_path.read_text())

    # The pre-registered verdict fields.
    for key in (
        "baseline",
        "challenger",
        "held_out_n_docs",
        "held_out_n_bytes",
        "compute_flops",
        "bpb_delta",
        "verdict",
        "falsifier",
        "falsifier_confirmed",
        "kill_criterion",
        "kill_triggered",
        "config",
    ):
        assert key in payload, key
    assert payload["verdict"] in {"climbmix_wins", "keep_fineweb_edu"}
    assert payload["falsifier_confirmed"] == (not payload["kill_triggered"])
    assert payload["held_out_n_docs"] == 4

    for arm in ("baseline", "challenger"):
        entry = payload[arm]
        assert math.isfinite(entry["val_bpb"]) and entry["val_bpb"] > 0
        assert entry["core"] is not None  # --core in toy mode = the fixture MC task
        assert entry["overlap_rate"] == 0.0  # toy held-out is excluded by construction
        assert entry["n_tokens"] == payload["config"]["steps"] * 8 * 64  # iso-FLOP

    # Incremental per-arm persistence fired before the verdict existed.
    for arm in ("fineweb_edu", "climbmix"):
        arm_json = json.loads((out / f"{arm}_result.json").read_text())
        assert arm_json["arm"] == arm
        assert (out / arm / "tokenizer.json").exists()
        assert (out / arm / "build_stats.json").exists()
        assert list((out / arm).glob("shard_*.bin"))


def test_toy_resume_adopts_staged_shards(tmp_path: Path) -> None:
    """A second run pointing at the first run's shard dirs skips staging (the banked-shards /
    resume path) and still logs the overlap rate from build_stats.json."""
    mod = _f12()
    out = tmp_path / "f12"
    mod.main([*_TOY_ARGS, "--out-dir", str(out)])

    out2 = tmp_path / "f12_resume"
    mod.main(
        [
            *_TOY_ARGS,
            "--out-dir",
            str(out2),
            "--fineweb-data-dir",
            str(out / "fineweb_edu"),
            "--climbmix-data-dir",
            str(out / "climbmix"),
        ]
    )
    payload = json.loads((out2 / "f12_verdict.json").read_text())
    for arm in ("baseline", "challenger"):
        assert payload[arm]["overlap_rate"] == 0.0  # recovered from build_stats.json, not None
    # No new shards were staged under the resume run's out-dir.
    assert not list(out2.glob("*/shard_*.bin"))
    assert payload["verdict"] in {"climbmix_wins", "keep_fineweb_edu"}
