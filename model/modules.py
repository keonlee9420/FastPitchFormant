import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

from .blocks import (
    LinearNorm,
    Conv1DBlock,
    FFTBlock,
)
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class TextEncoder(nn.Module):
    """ Text Encoder """

    def __init__(self, config):
        super(TextEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=0
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Spectrogram Decoder With Iterative Mel Prediction """

    def __init__(self, preprocess_config, model_config):
        super(Decoder, self).__init__()

        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        n_layers = model_config["transformer"]["decoder_layer"]
        n_head = model_config["transformer"]["decoder_head"]
        d_k = d_v = (
            model_config["transformer"]["decoder_hidden"]
            // model_config["transformer"]["decoder_head"]
        )
        d_model = model_config["transformer"]["decoder_hidden"]
        d_inner = model_config["transformer"]["conv_filter_size"]
        kernel_size = model_config["transformer"]["conv_kernel_size"]
        dropout = model_config["transformer"]["decoder_dropout"]

        self.n_mel_channels = n_mel_channels
        self.max_seq_len = model_config["max_seq_len"]
        self.d_model = d_model

        self.n_layers = n_layers
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_layer_1 = LinearNorm(
            d_model*2, n_mel_channels*2
        )
        self.fc_layers = nn.ModuleList(
            [
                LinearNorm(
                    d_model, n_mel_channels
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, formant_hidden, excitation_hidden, mask):

        mel_iters = list()
        max_len = formant_hidden.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- FC 1
        f_mel, e_mel = torch.split(self.fc_layer_1(torch.cat([formant_hidden, excitation_hidden], dim=-1)), \
            self.n_mel_channels, dim=-1)
        mel_iters.append(f_mel + e_mel)

        # -- FC 2, 3
        dec_output = formant_hidden + excitation_hidden
        for i, (dec_layer, linear) in enumerate(zip(self.layer_stack, self.fc_layers)):
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            mel_iters.append(
                linear(dec_output).masked_fill(mask.unsqueeze(-1), 0)
            )

        return mel_iters, mask


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        assert self.pitch_feature_level == "phoneme_level" # should be phoneme level

        d_model = model_config["transformer"]["encoder_hidden"]
        kernel_size = model_config["variance_embedding"]["kernel_size"]
        self.pitch_embedding = Conv1DBlock(
            1, d_model, kernel_size
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(target.unsqueeze(-1))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(prediction.unsqueeze(-1))
        return prediction, embedding

    def upsample(self, x, mel_mask, max_len, log_duration_prediction=None, duration_target=None, d_control=1.0):
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)
        return x, duration_rounded, mel_len, mel_mask

    def forward(
        self,
        x,
        speaker_embedding,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        duration_target=None,
        p_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, src_mask, p_control
        )

        if speaker_embedding is not None:
            pitch_embedding = pitch_embedding + speaker_embedding

        x, duration_rounded, mel_len, mel_mask = self.upsample(
            torch.cat([x, pitch_embedding], dim=-1), mel_mask, max_len, \
                log_duration_prediction=log_duration_prediction, duration_target=duration_target, d_control=d_control
        )

        text_hidden, pitch_hidden = torch.split(x, x.shape[-1]//2, dim=-1)

        return (
            text_hidden,
            pitch_hidden,
            pitch_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class Generator(nn.Module):
    """ Formant, Excitation Generator """

    def __init__(self, config, query_projection=False):
        super(Generator, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["generator_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.query_projection = query_projection
        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout, query_projection=query_projection
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, hidden, mask, hidden_query=None):

        if self.query_projection:
            assert hidden_query is not None, "Query should be given for the Excitation Generator."

        batch_size, max_len = hidden.shape[0], hidden.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and hidden.shape[1] > self.max_seq_len:
            output = hidden + get_sinusoid_encoding_table(
                hidden.shape[1], self.d_model
            )[: hidden.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                hidden.device
            )
        else:
            output = hidden + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for i, enc_layer in enumerate(self.layer_stack):
            output, enc_slf_attn = enc_layer(
                output, mask=mask, slf_attn_mask=slf_attn_mask, hidden_query=hidden_query if i==0 else None
            )

        return output


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
