"""Analyze the .npz dump from probe_decent_dp_features.py.

Outputs (under same dir as input .npz):
- {stem}_summary.txt       per-arm cosine-sim/L2 stats + ranks against cube positions
- {stem}_features_pca.png  3-panel PCA scatter (one per arm), color = seed idx
- {stem}_actions.png       N×Da×Ta overlay (one row per arm)
- {stem}_pairwise.png      pairwise feature cos-sim heatmaps + action L2 heatmaps per arm
- {stem}_thumbnails.png    head_cam thumbnails per (seed, agent) for visual sanity

Also exposes `linear_probe_features_to_cubes(npz_path, out_json)` for A7:
fits cube_xyz_concat = W @ feature with leave-one-out CV per-arm and reports
R^2 + permutation p-value (1000-sample null).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pairwise_cos(F):
    n = np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    return (F @ F.T) / (n @ n.T)


def pairwise_l2(X):
    return np.linalg.norm(X[:, None] - X[None, :], axis=-1)


def pca_2d(F):
    Fc = F - F.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    return Fc @ Vt[:2].T, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    npz = np.load(args.inp, allow_pickle=True)
    seeds = npz['seeds']
    feats = npz['features']     # (N, A, F)
    acts = npz['actions']        # (N, A, Ta, Da)
    images = npz['images']      # (N, A, 3, 60, 80)
    cubes = npz['cubes']        # (N, 3, 3)
    qpos = npz['qpos']
    ckpt_paths = npz['ckpt_paths']
    N, A, Fdim = feats.shape
    Ta, Da = acts.shape[2], acts.shape[3]

    inp = Path(args.inp)
    outdir = Path(args.outdir) if args.outdir else inp.parent
    stem = inp.stem

    summary_lines = []
    summary_lines.append(f"# probe analysis  N={N} arms={A} feat_dim={Fdim} action=({Ta},{Da})")
    summary_lines.append(f"  ckpts: {[str(p) for p in ckpt_paths]}")
    summary_lines.append(f"  cube xy std (m):")
    for ci, cn in enumerate(['cubeA', 'cubeB', 'cubeC']):
        s = np.nanstd(cubes[:, ci, :2], axis=0)
        summary_lines.append(f"    {cn}: dx={s[0]:.4f} dy={s[1]:.4f}")
    summary_lines.append("")

    # Per-arm pairwise stats
    summary_lines.append(f"{'arm':>3}  {'feat cos μ':>10} {'min':>8} {'max':>8}  {'feat var/F':>10}  "
                         f"{'act L2 μ':>10} {'min':>8} {'max':>8}  {'act var/T':>10}  {'corr(F,Cube)':>13}")
    cube_flat = cubes.reshape(N, -1)
    for aid in range(A):
        F = feats[:, aid]
        cs = pairwise_cos(F)
        cs_off = cs[np.triu_indices(N, k=1)]
        feat_var_per_dim = F.var(axis=0).mean()

        act_chunk = acts[:, aid].reshape(N, -1)
        l2 = pairwise_l2(act_chunk)
        l2_off = l2[np.triu_indices(N, k=1)]
        act_var_per_t = acts[:, aid].var(axis=0).mean()

        # correlation between feature distance and cube position distance
        cube_d = pairwise_l2(cube_flat)[np.triu_indices(N, k=1)]
        feat_d = (1 - cs)[np.triu_indices(N, k=1)]
        if cube_d.std() > 0 and feat_d.std() > 0:
            corr_fc = float(np.corrcoef(feat_d, cube_d)[0, 1])
        else:
            corr_fc = float('nan')

        summary_lines.append(
            f"{aid:>3}  {cs_off.mean():>10.4f} {cs_off.min():>8.4f} {cs_off.max():>8.4f}  "
            f"{feat_var_per_dim:>10.4e}  "
            f"{l2_off.mean():>10.4f} {l2_off.min():>8.4f} {l2_off.max():>8.4f}  "
            f"{act_var_per_t:>10.4e}  {corr_fc:>13.4f}"
        )

    summary_lines.append("")
    summary_lines.append("interpretation:")
    summary_lines.append("  feat cos μ ≈ 1.0  → encoder collapse")
    summary_lines.append("  feat cos μ < 0.95 + act L2 μ ≈ 0  → encoder fine, policy ignores obs")
    summary_lines.append("  feat cos μ < 0.95 + corr(F,Cube) ≈ 0 → encoder discriminates but not on cube position")
    summary_lines.append("  feat cos μ < 0.95 + corr(F,Cube) > 0 + act L2 μ small → policy under-attends to encoder")
    print("\n".join(summary_lines))

    summary_path = outdir / f"{stem}_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nwrote {summary_path}")

    # PCA scatter per arm
    fig, axes = plt.subplots(1, A, figsize=(5*A, 4.5), squeeze=False)
    for aid in range(A):
        F = feats[:, aid]
        proj, S = pca_2d(F)
        sc = axes[0, aid].scatter(proj[:, 0], proj[:, 1], c=np.arange(N), cmap='tab20', s=80)
        for i in range(N):
            axes[0, aid].annotate(str(seeds[i]), (proj[i, 0], proj[i, 1]),
                                  fontsize=7, ha='left', va='bottom')
        axes[0, aid].set_title(f"arm{aid} feat PCA  (sv ratio {S[0]/S[-1]:.2f})")
        axes[0, aid].set_xlabel('PC1'); axes[0, aid].set_ylabel('PC2')
        axes[0, aid].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_features_pca.png", dpi=120)
    plt.close(fig)

    # Pairwise heatmaps
    fig, axes = plt.subplots(2, A, figsize=(4.5*A, 8), squeeze=False)
    for aid in range(A):
        F = feats[:, aid]
        cs = pairwise_cos(F)
        im0 = axes[0, aid].imshow(cs, vmin=cs.min(), vmax=1.0, cmap='viridis')
        axes[0, aid].set_title(f"arm{aid} feat cos-sim")
        plt.colorbar(im0, ax=axes[0, aid], fraction=0.046)

        l2 = pairwise_l2(acts[:, aid].reshape(N, -1))
        im1 = axes[1, aid].imshow(l2, cmap='magma')
        axes[1, aid].set_title(f"arm{aid} action L2")
        plt.colorbar(im1, ax=axes[1, aid], fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_pairwise.png", dpi=120)
    plt.close(fig)

    # Action chunk overlay per arm — show all N chunks for each joint dim
    fig, axes = plt.subplots(A, Da, figsize=(2.0*Da, 2.4*A), squeeze=False, sharex=True)
    cmap = plt.get_cmap('tab20')
    for aid in range(A):
        for d in range(Da):
            for i in range(N):
                axes[aid, d].plot(np.arange(Ta), acts[i, aid, :, d], color=cmap(i / max(N-1, 1)),
                                  alpha=0.7, lw=1.0)
            axes[aid, d].grid(True, alpha=0.3)
            if aid == 0:
                axes[aid, d].set_title(f"act dim {d}", fontsize=9)
            if d == 0:
                axes[aid, d].set_ylabel(f"arm{aid}")
    fig.suptitle("predicted action chunks across seeds (overlay)")
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_actions.png", dpi=120)
    plt.close(fig)

    # Thumbnails grid
    fig, axes = plt.subplots(A, N, figsize=(1.5*N, 1.5*A), squeeze=False)
    for aid in range(A):
        for i in range(N):
            img = np.moveaxis(images[i, aid], 0, -1)
            axes[aid, i].imshow(np.clip(img, 0, 1))
            axes[aid, i].set_xticks([]); axes[aid, i].set_yticks([])
            if aid == 0:
                axes[aid, i].set_title(str(seeds[i]), fontsize=7)
            if i == 0:
                axes[aid, i].set_ylabel(f"arm{aid}")
    fig.suptitle("head_cam thumbnails (input to encoder)")
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_thumbnails.png", dpi=120)
    plt.close(fig)

    print(f"wrote PCA, pairwise, actions, thumbnails under {outdir}")


# -------------------------------------------------------------
# A7 — leave-one-out linear probe + permutation null
# -------------------------------------------------------------

def _loo_r2(X: np.ndarray, Y: np.ndarray, ridge_lambda: float = 1e-3) -> float:
    """Leave-one-out R^2 of ridge regression Y = X @ W + b.

    Uses Hat-matrix shortcut for the train-side Gram inverse (closed form per LOO),
    but for simplicity here we just fit n times — n is tiny (<= 32).
    Returns multivariate uniform-average R^2 across output dims.
    """
    N, D = X.shape
    Yhat = np.zeros_like(Y)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        Xt = X[mask]
        Yt = Y[mask]
        # ridge: (X^T X + λI)^-1 X^T Y
        XtX = Xt.T @ Xt + ridge_lambda * np.eye(D)
        W = np.linalg.solve(XtX, Xt.T @ Yt)
        b = Yt.mean(axis=0) - X[mask].mean(axis=0) @ W
        Yhat[i] = X[i] @ W + b
    # uniform-average R^2
    ss_res = ((Y - Yhat) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
    r2_per_dim = 1.0 - ss_res / ss_tot
    return float(r2_per_dim.mean())


def linear_probe_features_to_cubes(features_npz_path: str,
                                    out_json: str,
                                    n_perm: int = 1000,
                                    ridge_lambda: float = 1e-3,
                                    rng_seed: int = 0) -> dict:
    """A7: fit cube_xyz_concat = W @ feature with leave-one-out CV per-arm.

    Reports R^2 plus a permutation-bootstrap null (shuffled labels, n_perm
    samples) and an empirical p-value.

    Returns the result dict and writes it as JSON to out_json.
    """
    npz = np.load(features_npz_path, allow_pickle=True)
    feats = npz['features']  # (N, A, F) per-arm probes
    cubes = npz['cubes']     # (N, 3, 3)
    if feats.ndim == 2:
        # joint probe: (N, F) — reshape to (N, 1, F)
        feats = feats[:, None, :]
    N, A, F = feats.shape
    Y = cubes.reshape(N, -1)  # (N, 9) -- cubeA xyz, cubeB xyz, cubeC xyz

    rng = np.random.default_rng(rng_seed)
    arms_results = {}
    for aid in range(A):
        X = feats[:, aid].astype(np.float64)
        # center features per-dim
        X = X - X.mean(axis=0, keepdims=True)
        r2 = _loo_r2(X, Y.astype(np.float64), ridge_lambda=ridge_lambda)

        null_r2s = []
        for _ in range(n_perm):
            perm = rng.permutation(N)
            null_r2s.append(_loo_r2(X[perm], Y.astype(np.float64), ridge_lambda=ridge_lambda))
        null_r2s = np.asarray(null_r2s)
        # right-tail p-value
        p_value = float((null_r2s >= r2).mean())
        arms_results[f'arm{aid}'] = {
            'R2_loo': r2,
            'null_R2_mean': float(null_r2s.mean()),
            'null_R2_std': float(null_r2s.std()),
            'null_R2_p95': float(np.quantile(null_r2s, 0.95)),
            'p_value': p_value,
            'N': int(N),
            'F': int(F),
        }
        print(f"  arm{aid}: R2_loo={r2:.4f}  null μ={null_r2s.mean():.4f} ± {null_r2s.std():.4f}  "
              f"p95={np.quantile(null_r2s, 0.95):.4f}  p={p_value:.4f}", flush=True)

    out = {
        'features_npz': str(features_npz_path),
        'n_perm': int(n_perm),
        'ridge_lambda': float(ridge_lambda),
        'rng_seed': int(rng_seed),
        'arms': arms_results,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_json}")
    return out


def _cli_linear_probe():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-npz', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--ridge-lambda', type=float, default=1e-3)
    ap.add_argument('--rng-seed', type=int, default=0)
    args = ap.parse_args()
    linear_probe_features_to_cubes(
        args.features_npz, args.out_json,
        n_perm=args.n_perm, ridge_lambda=args.ridge_lambda, rng_seed=args.rng_seed,
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'linear-probe':
        # subcommand: analyze_decent_dp_features.py linear-probe --features-npz ... --out-json ...
        sys.argv.pop(1)
        _cli_linear_probe()
    else:
        main()
