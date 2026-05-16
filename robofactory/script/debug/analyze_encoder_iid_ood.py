"""Analyze per-seed encoder features (NPZ from probe_decent_dp_features) for
IID vs OOD separation.

Inputs:
    --npz   Path to NPZ produced by probe_decent_dp_features.probe_collapse_features
    --iid-seeds   Space-separated list of seeds treated as IID (training-distribution)
    --ood-seeds   Space-separated list of seeds treated as OOD (held-out test seeds)
    --out-dir     Output directory for PNG + JSON

For each arm, compute over the encoder-pooled feature distribution:
    intra-IID cos-sim
    intra-OOD cos-sim
    cross  IID-OOD cos-sim
Also report L2 distance equivalents. Plot 3-panel histogram and dump summary JSON.

Decision rule (encoder *not* the bottleneck):
    cross IID-OOD distribution overlaps intra-IID and intra-OOD distributions
    → encoder is treating OOD as in-manifold; failure is downstream.

Decision rule (encoder *is* the bottleneck):
    cross distribution is significantly LEFT-shifted (lower cos-sim,
    higher L2) relative to either intra distribution
    → encoder put IID and OOD on different manifolds; downstream UNet
    sees OOD features it never saw during training.

A third pathological case (collapse): both intra distributions concentrate
near 1.0 cos-sim → encoder is mapping everything to one vector regardless
of input.
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pairwise_cossim(F):
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    F_n = F / (norms + 1e-9)
    return F_n @ F_n.T


def pairwise_l2(F):
    return np.linalg.norm(F[:, None] - F[None, :], axis=-1)


def analyze(npz_path, iid_seeds, ood_seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(npz_path, allow_pickle=True)
    seeds = z["seeds"].tolist()
    feats = z["features"]  # (N, A, F)
    n_seeds, n_arms, feat_dim = feats.shape
    print(f"loaded {n_seeds} seeds, {n_arms} arms, feat_dim={feat_dim}")
    print(f"seeds in NPZ: {seeds}")

    iid_idx = [seeds.index(s) for s in iid_seeds if s in seeds]
    ood_idx = [seeds.index(s) for s in ood_seeds if s in seeds]
    missing_iid = set(iid_seeds) - set(seeds)
    missing_ood = set(ood_seeds) - set(seeds)
    if missing_iid:
        print(f"WARN: IID seeds in NPZ missing: {sorted(missing_iid)}")
    if missing_ood:
        print(f"WARN: OOD seeds in NPZ missing: {sorted(missing_ood)}")

    summary = {
        "npz": os.path.abspath(npz_path),
        "iid_seeds_used": [seeds[i] for i in iid_idx],
        "ood_seeds_used": [seeds[i] for i in ood_idx],
        "feat_dim": int(feat_dim),
        "n_arms": int(n_arms),
        "per_arm": [],
    }

    fig, axes = plt.subplots(n_arms, 2, figsize=(12, 4 * n_arms), squeeze=False)

    for aid in range(n_arms):
        F = feats[:, aid, :]
        F_iid = F[iid_idx]
        F_ood = F[ood_idx]

        cs_iid = pairwise_cossim(F_iid)
        cs_ood = pairwise_cossim(F_ood)
        F_iid_n = F_iid / (np.linalg.norm(F_iid, axis=1, keepdims=True) + 1e-9)
        F_ood_n = F_ood / (np.linalg.norm(F_ood, axis=1, keepdims=True) + 1e-9)
        cs_cross = F_iid_n @ F_ood_n.T

        l2_iid = pairwise_l2(F_iid)
        l2_ood = pairwise_l2(F_ood)
        l2_cross = np.linalg.norm(F_iid[:, None] - F_ood[None, :], axis=-1)

        iu_iid = np.triu_indices(len(iid_idx), k=1)
        iu_ood = np.triu_indices(len(ood_idx), k=1)
        cs_iid_vals = cs_iid[iu_iid]
        cs_ood_vals = cs_ood[iu_ood]
        cs_cross_vals = cs_cross.flatten()
        l2_iid_vals = l2_iid[iu_iid]
        l2_ood_vals = l2_ood[iu_ood]
        l2_cross_vals = l2_cross.flatten()

        arm_stats = {
            "arm": aid,
            "cos_sim": {
                "intra_iid": {"mean": float(cs_iid_vals.mean()), "std": float(cs_iid_vals.std()),
                              "min": float(cs_iid_vals.min()), "max": float(cs_iid_vals.max())},
                "intra_ood": {"mean": float(cs_ood_vals.mean()), "std": float(cs_ood_vals.std()),
                              "min": float(cs_ood_vals.min()), "max": float(cs_ood_vals.max())},
                "cross_iid_ood": {"mean": float(cs_cross_vals.mean()), "std": float(cs_cross_vals.std()),
                                  "min": float(cs_cross_vals.min()), "max": float(cs_cross_vals.max())},
            },
            "l2": {
                "intra_iid": {"mean": float(l2_iid_vals.mean()), "std": float(l2_iid_vals.std()),
                              "min": float(l2_iid_vals.min()), "max": float(l2_iid_vals.max())},
                "intra_ood": {"mean": float(l2_ood_vals.mean()), "std": float(l2_ood_vals.std()),
                              "min": float(l2_ood_vals.min()), "max": float(l2_ood_vals.max())},
                "cross_iid_ood": {"mean": float(l2_cross_vals.mean()), "std": float(l2_cross_vals.std()),
                                  "min": float(l2_cross_vals.min()), "max": float(l2_cross_vals.max())},
            },
        }
        # Verdict heuristic
        m_intra = 0.5 * (cs_iid_vals.mean() + cs_ood_vals.mean())
        m_cross = cs_cross_vals.mean()
        gap = m_intra - m_cross  # higher gap → IID and OOD live on different manifolds
        if cs_iid_vals.mean() > 0.995 and cs_ood_vals.mean() > 0.995:
            arm_stats["verdict"] = "collapsed_encoder"
        elif gap > 0.05:
            arm_stats["verdict"] = "iid_ood_separated"  # plausibly the bottleneck
        else:
            arm_stats["verdict"] = "no_separation"      # encoder is not the bottleneck
        arm_stats["cossim_intra_minus_cross"] = float(gap)
        summary["per_arm"].append(arm_stats)

        ax = axes[aid, 0]
        bins = np.linspace(min(cs_iid_vals.min(), cs_ood_vals.min(), cs_cross_vals.min()), 1.0, 30)
        ax.hist(cs_iid_vals, bins=bins, alpha=0.5, label=f"intra-IID (n={len(cs_iid_vals)})", color="C0")
        ax.hist(cs_ood_vals, bins=bins, alpha=0.5, label=f"intra-OOD (n={len(cs_ood_vals)})", color="C1")
        ax.hist(cs_cross_vals, bins=bins, alpha=0.5, label=f"cross IID-OOD (n={len(cs_cross_vals)})", color="C3")
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("count")
        ax.set_title(f"arm{aid} cos-sim — gap={gap:+.4f} verdict={arm_stats['verdict']}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[aid, 1]
        all_l2 = np.concatenate([l2_iid_vals, l2_ood_vals, l2_cross_vals])
        bins = np.linspace(all_l2.min(), all_l2.max(), 30)
        ax.hist(l2_iid_vals, bins=bins, alpha=0.5, label="intra-IID", color="C0")
        ax.hist(l2_ood_vals, bins=bins, alpha=0.5, label="intra-OOD", color="C1")
        ax.hist(l2_cross_vals, bins=bins, alpha=0.5, label="cross", color="C3")
        ax.set_xlabel("L2 distance")
        ax.set_ylabel("count")
        ax.set_title(f"arm{aid} L2")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "encoder_iid_ood_hist.png")
    plt.savefig(png_path, dpi=110)
    plt.close()

    summary_path = os.path.join(out_dir, "encoder_iid_ood_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved {png_path}")
    print(f"saved {summary_path}")
    for arm in summary["per_arm"]:
        print(f"  arm{arm['arm']}: verdict={arm['verdict']}  "
              f"cossim_gap={arm['cossim_intra_minus_cross']:+.4f}  "
              f"intra-IID={arm['cos_sim']['intra_iid']['mean']:.4f}  "
              f"intra-OOD={arm['cos_sim']['intra_ood']['mean']:.4f}  "
              f"cross={arm['cos_sim']['cross_iid_ood']['mean']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--iid-seeds", type=int, nargs="+", required=True)
    ap.add_argument("--ood-seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    analyze(args.npz, args.iid_seeds, args.ood_seeds, args.out_dir)


if __name__ == "__main__":
    main()
