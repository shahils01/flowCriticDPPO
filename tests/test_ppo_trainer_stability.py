import torch

from ppo.algorithms.ppo.ppo_trainer import PPOTrainer


def test_value_loss_active_masks_average_over_particles():
    trainer = object.__new__(PPOTrainer)
    trainer.clip_param = 0.2
    trainer.value_normalizer = None
    trainer._use_huber_loss = False
    trainer._use_clipped_value_loss = False
    trainer._use_value_active_masks = True
    trainer.huber_delta = 10.0
    values = torch.ones(2, 4)
    old_values = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    active_masks = torch.tensor([[1.0], [0.0]])

    loss = PPOTrainer.scalar_value_loss(
        trainer,
        values=values,
        old_values=old_values,
        value_targets=returns,
        active_masks=active_masks,
    )

    assert torch.allclose(loss, torch.tensor(0.5))
