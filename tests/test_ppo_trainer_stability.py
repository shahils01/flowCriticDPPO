from types import SimpleNamespace

import torch

from ppo.algorithms.ppo.ppo_trainer import PPOTrainer


def test_value_loss_active_masks_average_over_particles():
    trainer = SimpleNamespace(
        clip_param=0.2,
        _use_valuenorm=False,
        value_normalizer=None,
        _use_huber_loss=False,
        _use_clipped_value_loss=False,
        _use_value_active_masks=True,
        huber_delta=10.0,
    )
    values = torch.ones(2, 4)
    old_values = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    active_masks = torch.tensor([[1.0], [0.0]])

    loss = PPOTrainer.cal_value_loss(
        trainer,
        values=values,
        value_preds_batch=old_values,
        return_batch=returns,
        active_masks_batch=active_masks,
    )

    assert torch.allclose(loss, torch.tensor(0.5))
