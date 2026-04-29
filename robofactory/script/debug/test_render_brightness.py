"""Test that env-reset rendering brightness matches paper training data within tolerance."""
import sys, os, numpy as np
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')
import gymnasium as gym
from robofactory.tasks import *
from PIL import Image

PAPER_FRAME = '/iris/u/mikulrai/data/debug_assets/train_frame_traj0_head_cam.png'
TARGET_RGB = np.asarray(Image.open(PAPER_FRAME)).reshape(-1, 3).mean(0)  # ~[160, 117, 94]
print(f"TARGET (paper) mean RGB = {TARGET_RGB.round(2).tolist()}")
print(f"VK_ICD_FILENAMES (pre-sapien) = {os.environ.get('VK_ICD_FILENAMES', '<unset>')}")
print(f"VK_LOADER_DEBUG  = {os.environ.get('VK_LOADER_DEBUG', '<unset>')}")
# Importing sapien runs _vulkan_tricks which may overwrite VK_ICD_FILENAMES
import sapien  # noqa: E402
print(f"VK_ICD_FILENAMES (post-sapien) = {os.environ.get('VK_ICD_FILENAMES', '<unset>')}")
print(f"SAPIEN_VULKAN_LIBRARY_PATH = {os.environ.get('SAPIEN_VULKAN_LIBRARY_PATH', '<unset>')}")
print(f"__EGL_VENDOR_LIBRARY_FILENAMES = {os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES', '<unset>')}")

def render_seed_k(seed=0, shader='default', enable_shadow=False):
    env = gym.make('PickMeat-rf', config='configs/table/pick_meat.yaml',
                   obs_mode='rgb', control_mode='pd_joint_pos',
                   render_mode='sensors', num_envs=1, sim_backend='cpu',
                   enable_shadow=enable_shadow,
                   sensor_configs=dict(shader_pack=shader))
    obs, _ = env.reset(seed=seed)
    rgb = obs['sensor_data']['head_camera']['rgb']
    arr = (rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.asarray(rgb)).squeeze(0)
    env.close()
    return arr.reshape(-1, 3).mean(0), arr

results = {}
for shader in ['default', 'rt-fast', 'rt']:
    try:
        m, arr = render_seed_k(seed=0, shader=shader)
        diff = float(np.abs(m - TARGET_RGB).mean())
        print(f"shader={shader:>10s}: mean RGB={m.round(1).tolist()}  diff vs paper={diff:.2f}")
        results[shader] = (m, arr, diff)
        # Save image for visual inspection
        out_path = f"/iris/u/mikulrai/data/debug_assets/eval_brightness_{shader}.png"
        Image.fromarray(arr.astype(np.uint8)).save(out_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"shader={shader:>10s}: ERROR {type(e).__name__}: {e}")
        results[shader] = None

PASS_TOL = 5.0
if results.get('default') is not None:
    m, _, diff = results['default']
    if diff < PASS_TOL:
        print(f"PASS: rendering matches paper to within {PASS_TOL} RGB units (got {diff:.2f})")
        sys.exit(0)
    else:
        print(f"FAIL: rendering diff {diff:.2f} exceeds {PASS_TOL} units. Need further fix.")
        sys.exit(1)
else:
    print(f"FAIL: shader='default' errored")
    sys.exit(1)
