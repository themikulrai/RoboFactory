"""Unit + cross-driver tests for the single seed-convention module (PR4).

Run the seed-convention slice with:  ``pytest -k eval_seeds``

Covers:
  - per-pool values + pool_sha stability (the pin must never silently drift);
  - resolve_seeds provenance + the train-overlap guard;
  - CROSS-DRIVER: all FOUR eval drivers resolve canonical_env_60 to the SAME
    list and none retains the deleted in-driver transform (true seed pairing);
  - the pinned runs/ files match the module (the pin == the source of truth);
  - lint rule G (60-seed launchers must --seed-pool or use a pinned env-seed file).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from robofactory.utils import eval_seeds as es


REPO_ROOT = Path(__file__).resolve().parents[1]  # robofactory/
PROJECT_ROOT = REPO_ROOT.parent
RUNS_DIR = Path("/iris/u/mikulrai/runs")

# The four eval drivers that must consume the SAME resolved env seeds (PR4).
DRIVER_PATHS = {
    "eval_pi05": REPO_ROOT / "policy" / "openpi_pi05" / "eval_pi05.py",
    "eval_decent_pi05": REPO_ROOT / "policy" / "openpi_pi05" / "eval_decent_pi05.py",
    "eval_hier_lb": REPO_ROOT / "policy" / "openpi_pi05" / "eval_hierarchical_lift_barrier.py",
    "eval_multi_dp": REPO_ROOT / "policy" / "Diffusion-Policy" / "eval_multi_dp.py",
    "eval_joint_dp": REPO_ROOT / "policy" / "Diffusion-Policy" / "eval_joint_dp.py",
}


# --------------------------------------------------------------------------- pools
class TestPoolValues:
    def test_train_datagen(self):
        p = es.get_pool("train_datagen")
        assert p == list(range(0, 183))
        assert p[0] == 0 and p[-1] == 182 and len(p) == 183

    def test_canonical_env_60(self):
        p = es.get_pool("canonical_env_60")
        assert p == [b * 100_000 for b in range(100, 160)]
        assert p[0] == 10_000_000 and p[-1] == 15_900_000 and len(p) == 60
        # The historical x100_000 transform is folded in: the pool already IS the
        # final env seeds, so no driver should multiply again.
        assert all(s % 100_000 == 0 for s in p)

    def test_fresh_ood_60(self):
        p = es.get_pool("fresh_ood_60")
        assert p == list(range(20_000, 20_060))
        assert len(p) == 60

    def test_upstream_100(self):
        p = es.get_pool("upstream_100")
        assert p == list(range(1_000, 1_100))
        assert len(p) == 100

    def test_dp_legacy_60(self):
        p = es.get_pool("dp_legacy_60")
        assert p == list(range(10_000, 10_030)) + list(range(1_000, 1_030))
        assert len(p) == 60
        assert "dp_legacy_60" in es.DEPRECATED_POOLS

    def test_unknown_pool_raises(self):
        with pytest.raises(KeyError):
            es.get_pool("does_not_exist")

    def test_get_pool_returns_fresh_copy(self):
        a = es.get_pool("canonical_env_60")
        a.append(-1)
        b = es.get_pool("canonical_env_60")
        assert -1 not in b  # mutating the returned list must not corrupt the pool


# ----------------------------------------------------------------- pool_sha stability
class TestPoolSha:
    # These hashes PIN the pools. A change here means a pool's values changed —
    # which silently re-defines every "paired" comparison, so it must be deliberate.
    EXPECTED = {
        "train_datagen": es._sha_of_seeds(list(range(0, 183))),
        "canonical_env_60": "59643d0d7cdedc03330a2b341b3bd9248104cb5cbc803a22deed58bf2ca49f0a",
        "fresh_ood_60": es._sha_of_seeds(list(range(20_000, 20_060))),
        "upstream_100": es._sha_of_seeds(list(range(1_000, 1_100))),
        "dp_legacy_60": es._sha_of_seeds(list(range(10_000, 10_030)) + list(range(1_000, 1_030))),
    }

    def test_canonical_sha_is_pinned_literal(self):
        # Hard literal so a refactor that re-derives the pool can't move the pin.
        assert es.pool_sha("canonical_env_60") == self.EXPECTED["canonical_env_60"]

    def test_all_pool_shas_match(self):
        for name, want in self.EXPECTED.items():
            assert es.pool_sha(name) == want, name

    def test_sha_is_order_sensitive(self):
        a = es.seeds_sha([1, 2, 3])
        b = es.seeds_sha([3, 2, 1])
        assert a != b

    def test_sha_stable_across_calls(self):
        assert es.pool_sha("upstream_100") == es.pool_sha("upstream_100")


# ----------------------------------------------------------------- resolve + guard
class TestResolveSeeds:
    def test_resolve_pool(self):
        seeds, prov = es.resolve_seeds(pool="canonical_env_60")
        assert seeds == es.get_pool("canonical_env_60")
        assert prov["seed_pool"] == "canonical_env_60"
        assert prov["seed_pool_sha"] == es.pool_sha("canonical_env_60")
        assert prov["n_seeds"] == 60
        assert prov["allow_train"] is False

    def test_resolve_env_seeds_adhoc(self):
        seeds, prov = es.resolve_seeds(env_seeds="1000, 2000 3000")
        assert seeds == [1000, 2000, 3000]
        assert prov["seed_pool"] == "adhoc"
        assert prov["n_seeds"] == 3

    def test_resolve_requires_exactly_one(self):
        with pytest.raises(ValueError):
            es.resolve_seeds()
        with pytest.raises(ValueError):
            es.resolve_seeds(pool="canonical_env_60", env_seeds="1,2,3")

    def test_train_overlap_hard_fails(self):
        with pytest.raises(ValueError):
            es.resolve_seeds(pool="train_datagen")
        with pytest.raises(ValueError):
            es.resolve_seeds(env_seeds="5,6,7")  # 5/6/7 are datagen seeds

    def test_train_overlap_allowed_records_flag(self):
        seeds, prov = es.resolve_seeds(pool="train_datagen", allow_train=True)
        assert seeds == list(range(0, 183))
        assert prov["allow_train"] is True

    def test_held_out_pools_pass_guard(self):
        for name in ("canonical_env_60", "fresh_ood_60", "upstream_100", "dp_legacy_60"):
            es.assert_no_train_overlap(es.get_pool(name))  # must not raise

    def test_boundary_182_is_train_183_is_not(self):
        assert es.is_train_seed(182) is True
        assert es.is_train_seed(183) is False
        assert es.is_train_seed(0) is True
        with pytest.raises(ValueError):
            es.assert_no_train_overlap([182])
        es.assert_no_train_overlap([183])  # must not raise


# ----------------------------------------------------------------- pinned runs/ files
class TestPinnedFiles:
    @pytest.mark.skipif(not RUNS_DIR.exists(), reason="runs dir not mounted")
    def test_env_60_pin_matches_module(self):
        txt = (RUNS_DIR / "eval_seeds_env_60.txt").read_text()
        file_seeds = [int(x) for x in txt.split()]
        assert file_seeds == es.get_pool("canonical_env_60")

    @pytest.mark.skipif(not RUNS_DIR.exists(), reason="runs dir not mounted")
    def test_env_60_sha_pin_matches_file(self):
        p = RUNS_DIR / "eval_seeds_env_60.txt"
        want = hashlib.sha256(p.read_bytes()).hexdigest()
        got = (RUNS_DIR / "eval_seeds_env_60.txt.sha256").read_text().strip()
        assert got == want


# ----------------------------------------------------------------- CROSS-DRIVER
class TestCrossDriverSeedPairing:
    """All four drivers must resolve canonical_env_60 to the SAME env seeds and
    must NOT retain a per-driver transform. We assert this two ways: (a) the one
    resolver every driver calls is deterministic; (b) each driver source calls
    resolve_seeds and no longer contains the deleted x100_000 / *stride transform.
    Source-grep (not import) because importing a driver pulls in sapien/torch."""

    def test_resolver_is_single_valued(self):
        # The four drivers all delegate to resolve_seeds(pool=...). Resolving the
        # same pool name yields one canonical list -> true seed pairing.
        lists = [es.resolve_seeds(pool="canonical_env_60")[0] for _ in range(4)]
        assert all(l == lists[0] for l in lists)
        assert lists[0] == es.get_pool("canonical_env_60")

    def test_all_drivers_call_resolve_seeds(self):
        for name, path in DRIVER_PATHS.items():
            src = path.read_text()
            assert "resolve_seeds(" in src, f"{name} must call resolve_seeds (PR4)"

    def test_no_driver_retains_deleted_transform(self):
        # The historical transforms are DELETED: pi0.5 `seed * 100_000`, hier
        # `(seed+ep)*seed_stride`. None may survive as an active code path.
        bad_pi05 = re.compile(r'\*\s*100_000|100_000\s*\*|\*\s*100000')
        bad_stride = re.compile(r'\*\s*args\.seed_stride|seed_stride\s*\*')
        for name, path in DRIVER_PATHS.items():
            for ln in path.read_text().splitlines():
                code = ln.split("#", 1)[0]  # ignore comments (they document the deletion)
                assert not bad_pi05.search(code), f"{name}: leftover x100_000 transform: {ln!r}"
                assert not bad_stride.search(code), f"{name}: leftover *seed_stride transform: {ln!r}"

    def test_drivers_expose_seed_pool_flag(self):
        for name, path in DRIVER_PATHS.items():
            assert "seed_pool" in path.read_text(), f"{name} must expose --seed-pool"


# ----------------------------------------------------------------- lint rule G
class TestLintRuleG:
    def _good_pool_launcher(self, tmp_path: Path) -> Path:
        f = tmp_path / "good_pool_60seeds.sh"
        f.write_text(
            "#!/bin/bash\n"
            "source _lib/free_ports.sh\n"
            'read -r PORT0 <<<"$(free_ports 1)"\n'
            "python eval_pi05.py --seed-pool canonical_env_60 --port \"$PORT0\"\n"
        )
        return f

    def _good_pinfile_launcher(self, tmp_path: Path) -> Path:
        f = tmp_path / "good_pin_60seeds.sh"
        f.write_text(
            "#!/bin/bash\n"
            "SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_env_60.txt)\n"
            "python eval_pi05.py --seeds \"$SEEDS\" --port 12345\n"
        )
        return f

    def test_rule_g_clean_with_seed_pool(self, tmp_path: Path):
        from robofactory.utils.lint_eval_launchers import lint_one
        rpt = lint_one(self._good_pool_launcher(tmp_path))
        assert [x for x in rpt.findings if x.rule == "G"] == []

    def test_rule_g_clean_with_pinned_file(self, tmp_path: Path):
        from robofactory.utils.lint_eval_launchers import lint_one
        rpt = lint_one(self._good_pinfile_launcher(tmp_path))
        assert [x for x in rpt.findings if x.rule == "G"] == []

    def test_rule_g_flags_raw_seeds(self, tmp_path: Path):
        from robofactory.utils.lint_eval_launchers import lint_one
        f = tmp_path / "bad_raw_60seeds.sh"
        f.write_text(
            "#!/bin/bash\n"
            "SEEDS=$(paste -sd, /iris/u/mikulrai/runs/some_other_seeds.txt)\n"
            "python eval_pi05.py --seeds \"$SEEDS\" --port 12345\n"
        )
        rpt = lint_one(f)
        g = [x for x in rpt.findings if x.rule == "G"]
        assert len(g) == 1, f"expected one rule-G finding, got {rpt.findings}"

    def test_rule_g_flags_unknown_pool(self, tmp_path: Path):
        from robofactory.utils.lint_eval_launchers import lint_one
        f = tmp_path / "bad_pool_60seeds.sh"
        f.write_text(
            "#!/bin/bash\n"
            "python eval_pi05.py --seed-pool not_a_real_pool --port 12345\n"
        )
        rpt = lint_one(f)
        g = [x for x in rpt.findings if x.rule == "G"]
        assert len(g) == 1 and "not_a_real_pool" in g[0].message

    def test_rule_g_na_for_non_60seed_launcher(self, tmp_path: Path):
        from robofactory.utils.lint_eval_launchers import lint_one
        f = tmp_path / "smoke_3seeds.sh"  # not a 60-seed launcher
        f.write_text("#!/bin/bash\npython eval_pi05.py --seeds 1,2,3 --port 12345\n")
        rpt = lint_one(f)
        assert [x for x in rpt.findings if x.rule == "G"] == []

    def test_canonical_launchers_rule_g_clean(self):
        """Lint G must be CLEAN on the real canonical 60-seed launchers (spec)."""
        from robofactory.utils.lint_eval_launchers import lint_one
        launchers = sorted((REPO_ROOT / "scripts" / "canonical").glob("*_60seeds*.sh"))
        assert launchers, "no canonical 60-seed launchers found"
        offenders = []
        for p in launchers:
            rpt = lint_one(p)
            g = [x for x in rpt.findings if x.rule == "G"]
            if g:
                offenders.append((p.name, [x.message for x in g]))
        assert offenders == [], f"rule-G findings on canonical launchers: {offenders}"
