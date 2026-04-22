if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import time
import hydra
import torch
import wandb
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)
import pdb

class RobotWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # pdb.set_trace()

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        # import pdb; pdb.set_trace()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        try:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
        except Exception as e:
            print(f"[env_runner] init failed, skipping rollouts: {e}")
            env_runner = None

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # parameter count (one-shot to wandb.summary)
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.summary['params/total_millions'] = total_params / 1e6
            wandb.summary['params/trainable_millions'] = trainable_params / 1e6
            wandb.summary['params/current_agent_id'] = int(getattr(cfg, 'current_agent_id', 0))
        except Exception as e:
            print(f"[wandb] param-count logging failed: {e}")

        # dataset summary (one-shot to wandb.summary)
        try:
            action_arr = np.asarray(dataset.replay_buffer['action'])
            state_arr = np.asarray(dataset.replay_buffer['state'])
            head_shape = list(dataset.replay_buffer['head_camera'].shape)
            wandb.summary['dataset/n_train_episodes'] = int(dataset.train_mask.sum())
            wandb.summary['dataset/n_val_episodes'] = int((~dataset.train_mask).sum())
            wandb.summary['dataset/n_train_samples'] = len(dataset)
            wandb.summary['dataset/n_val_samples'] = len(val_dataset)
            wandb.summary['dataset/action_dim'] = int(action_arr.shape[-1])
            wandb.summary['dataset/state_dim'] = int(state_arr.shape[-1])
            wandb.summary['dataset/image_shape'] = head_shape[1:]
            wandb.summary['dataset/action_min'] = float(action_arr.min())
            wandb.summary['dataset/action_max'] = float(action_arr.max())
            wandb.summary['dataset/state_min'] = float(state_arr.min())
            wandb.summary['dataset/state_max'] = float(state_arr.max())
            wandb.summary['dataset/zarr_path'] = str(cfg.task.dataset.zarr_path)
        except Exception as e:
            print(f"[wandb] dataset summary logging failed: {e}")

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                epoch_start_time = time.time()
                epoch_samples = 0
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        # pdb.set_trace()
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer (also measure grad norm right before step)
                        grad_norm = None
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=float('inf')
                            ).item()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        try:
                            epoch_samples += int(batch['action'].shape[0])
                        except Exception:
                            pass
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        if grad_norm is not None:
                            step_log['grad_norm'] = grad_norm

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            wandb.log(step_log, step=self.global_step)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if env_runner is not None and (self.epoch % cfg.training.rollout_every) == 0:
                    try:
                        runner_log = env_runner.run(policy)
                    except Exception as e:
                        import traceback as _tb
                        _tb.print_exc()
                        runner_log = {
                            'test_mean_score': 0.0,
                            'rollout/enabled': 0,
                            'rollout/error': f'run_exception:{type(e).__name__}',
                        }
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                        # same diffusion sampling on one val batch for val action MSE
                        try:
                            for val_batch_raw in val_dataloader:
                                val_batch = dataset.postprocess(val_batch_raw, device)
                                v_obs = val_batch['obs']
                                v_gt = val_batch['action']
                                v_result = policy.predict_action(v_obs)
                                v_pred = v_result['action_pred']
                                v_mse = torch.nn.functional.mse_loss(v_pred, v_gt)
                                step_log['val_action_mse_error'] = v_mse.item()
                                del val_batch_raw, val_batch, v_obs, v_gt, v_result, v_pred, v_mse
                                break
                        except Exception as _e:
                            print(f"[val_action_mse] skipped: {_e}")
                
                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    self.save_checkpoint(f'checkpoints/{save_name}/{self.epoch + 1}.ckpt') # TODO
                
                # ========= eval end for this epoch ==========
                policy.train()

                # EMA decay snapshot
                if cfg.training.use_ema and ema is not None:
                    ema_decay = getattr(ema, 'decay', None)
                    if ema_decay is None:
                        ema_decay = float(cfg.ema.max_value)
                    try:
                        step_log['ema_decay'] = float(ema_decay)
                    except Exception:
                        pass

                # epoch wall-time + samples/sec
                epoch_elapsed = time.time() - epoch_start_time
                step_log['epoch_time_sec'] = epoch_elapsed
                if epoch_elapsed > 0 and epoch_samples > 0:
                    step_log['samples_per_sec'] = epoch_samples / epoch_elapsed

                # best-so-far via wandb.summary
                if not hasattr(self, '_best_metrics'):
                    self._best_metrics = {}
                for _key, _mode in [
                    ('val_loss', 'min'),
                    ('val_action_mse_error', 'min'),
                    ('train_action_mse_error', 'min'),
                    ('test_mean_score', 'max'),
                    ('rollout/success_rate', 'max'),
                ]:
                    if _key not in step_log:
                        continue
                    _val = step_log[_key]
                    try:
                        _val = float(_val)
                    except Exception:
                        continue
                    _best_key = f'best/{_key}'
                    _cur = self._best_metrics.get(_best_key)
                    _is_better = (
                        _cur is None
                        or (_mode == 'min' and _val < _cur)
                        or (_mode == 'max' and _val > _cur)
                    )
                    if _is_better:
                        self._best_metrics[_best_key] = _val
                        try:
                            wandb.summary[_best_key] = _val
                            wandb.summary[f'{_best_key}_epoch'] = self.epoch
                        except Exception:
                            pass

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                wandb.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1

        wandb.finish()


class BatchSampler:
    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False, seed: int = 0, drop_last: bool = True):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch

def create_dataloader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, persistent_workers: bool, seed: int = 0):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)
    def collate(x):
        assert len(x) == 1
        return x[0]
    dataloader = DataLoader(dataset, collate_fn=collate, sampler=batch_sampler, num_workers=num_workers, pin_memory=False, persistent_workers=persistent_workers)
    return dataloader

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
