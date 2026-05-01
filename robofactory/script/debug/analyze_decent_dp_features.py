"""Analyze the .npz dump from probe_decent_dp_features.py.

Outputs (under same dir as input .npz):
- {stem}_summary.txt       per-arm cosine-sim/L2 stats + ranks against cube positions
- {stem}_features_pca.png  3-panel PCA scatter (one per arm), color = seed idx
- {stem}_actions.png       N×Da×Ta overlay (one row per arm)
- {stem}_pairwise.png      pairwise feature cos-sim heatmaps + action L2 heatmaps per arm
- {stem}_thumbnails.png    head_cam thumbnails per (seed, agent) for visual sanity
"""
import argparse
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


if __name__ == '__main__':
    main()
