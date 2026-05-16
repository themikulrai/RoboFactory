"""Decode recorded RGB frames from an H5 trajectory to MP4.

This is the *training-time* view: what the planner saw when the demo was
recorded. No SAPIEN needed - just decode the stored uint8 tensors.
"""
import argparse
import os
import h5py
import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--traj', type=int, required=True)
    ap.add_argument('--cam', default='head_camera_global')
    ap.add_argument('--out-mp4', required=True)
    ap.add_argument('--fps', type=int, default=20)
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        tr = f[f'traj_{args.traj}']
        frames = tr['obs']['sensor_data'][args.cam]['rgb'][...]
        success = bool(tr['success'][-1])
        T = frames.shape[0]
    print(f'loaded {T} frames from {args.cam}, h5_success={success}')

    os.makedirs(os.path.dirname(args.out_mp4), exist_ok=True)
    with imageio.get_writer(args.out_mp4, fps=args.fps, codec='libx264',
                            quality=8, macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)
    print(f'wrote {args.out_mp4}')


if __name__ == '__main__':
    main()
