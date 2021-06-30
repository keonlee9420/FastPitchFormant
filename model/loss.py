import torch
import torch.nn as nn


class FastPitchFormantLoss(nn.Module):
    """ FastPitchFormant Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastPitchFormantLoss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.mse_loss_sum = nn.MSELoss(reduction='sum')
        self.mse_loss = nn.MSELoss()
        # self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            mel_lens_targets,
            _,
            pitch_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_iters,
            pitch_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        mel_targets.requires_grad = False
        mel_lens_targets.requires_grad = False

        pitch_predictions = pitch_predictions.masked_select(src_masks)
        pitch_targets = pitch_targets.masked_select(src_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = 0
        for mel_iter in mel_iters:
            mel_predictions = mel_iter.masked_select(mel_masks.unsqueeze(-1))
            mel_loss += self.mse_loss_sum(mel_predictions, mel_targets)
        mel_loss = (mel_loss / (self.n_mel_channels * mel_lens_targets)).mean()

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + duration_loss + pitch_loss
        )

        return (
            total_loss,
            mel_loss,
            pitch_loss,
            duration_loss,
        )
