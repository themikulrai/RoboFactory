"""Quick sanity check: load the qpos retrained ckpt, feed a zarr observation,
print the predicted action chunk. Verify the model produces sensible output."""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import numpy as np
import torch
import dill
import zarr
import hydra

from diffusion_policy.workspace.robotworkspace import RobotWorkspace


@torch.no_grad()
def main():
    for arm in (0, 1):
        ckpt = f'/iris/u/mikulrai/checkpoints/RoboFactory/TwoRobotsStackCube-rf_workspace_overfit1qpos_agent{arm}_150/2000.ckpt'
        zp = f'/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_overfit1qpos_agent{arm}_150.zarr'
        print(f'\n=== arm {arm} ===')
        print(f'ckpt: {ckpt}')
        print(f'zarr: {zp}')

        payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        ws = cls(cfg, output_dir=None)
        ws.load_payload(payload, exclude_keys=None, include_keys=None)
        pol = ws.ema_model if cfg.training.use_ema else ws.model
        pol.to('cuda:0').eval()

        zr = zarr.open(zp, mode='r')
        ee = np.asarray(zr['meta/episode_ends'])
        s = int(ee[12])  # episode 13 start
        # Build training obs window for t=10 (somewhere in motion)
        t = 10
        idx_lo = t - 2  # n_obs_steps=3
        head = np.asarray(zr['data/head_camera'][s+idx_lo:s+t+1], dtype=np.float32) / 255.0   # (3, 3, 224, 224)
        head_g = np.asarray(zr['data/head_camera_global'][s+idx_lo:s+t+1], dtype=np.float32) / 255.0
        state = np.asarray(zr['data/state'][s+idx_lo:s+t+1], dtype=np.float32)                # (3, 8)
        action_target = np.asarray(zr['data/action'][s+t], dtype=np.float32)                   # (8,)

        obs = {
            'head_cam': torch.from_numpy(head).unsqueeze(0).to('cuda:0'),       # (1, 3, 3, 224, 224)
            'head_cam_global': torch.from_numpy(head_g).unsqueeze(0).to('cuda:0'),
            'agent_pos': torch.from_numpy(state).unsqueeze(0).to('cuda:0'),     # (1, 3, 8)
        }
        out = pol.predict_action(obs)
        chunk = out['action'].cpu().numpy()                                     # (1, horizon, 8)
        pred = chunk[0, 0]                                                       # action at t (predicted)
        print(f'state[t] = {state[-1].round(4)}')
        print(f'recorded action[t] = {action_target.round(4)}')
        print(f'predicted chunk[0] = {pred.round(4)}')
        print(f'L2 = {float(np.linalg.norm(pred - action_target)):.4f}')
        print(f'chunk shape = {chunk.shape}  finite={np.all(np.isfinite(chunk))}')


if __name__ == '__main__':
    main()
