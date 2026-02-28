# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import copy
import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils

from paddlefleet.fp8 import FP8ColumnParallelLinear
from paddlefleet.fp8.utils import is_fp8_tensor
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.pipeline_parallel import NoPipelineParallel
from paddlefleet.tensor_parallel import ColumnParallelLinear
from paddlefleet.transformer.transformer_config import TransformerConfig


def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    x, y = x.double().numpy(), y.double().numpy()
    denominator = (x * x + y * y).sum()
    if denominator == 0:  # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class TestParallelMLP(unittest.TestCase):
    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=4096,
            intermediate_size=7168,
            use_bias=False,
            use_cpu_initialization=True,
        )

        set_seed(123)
        self.fp8_linear = FP8ColumnParallelLinear(
            self.config.hidden_size,
            self.config.intermediate_size,
            config=self.config,
            init_method=self.config.init_method,
        )

        paddle.amp.decorate(
            models=self.fp8_linear,
            level="O2",
            dtype="bfloat16",
        )
        self.fp8_linear.weight.main_grad = None

        set_seed(123)
        # self.fp32_linear = paddle.nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias_attr=False)
        self.fp32_linear = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.intermediate_size,
            init_method=self.config.init_method,
            bias=False,
            config=self.config,
            skip_bias_add=False,
            gather_output=False,
            tp_group=None,
        )

        self.acc_step = 4

    def test_utils(self):
        batch_size = 16384
        np_x = np.random.randn(batch_size, self.config.hidden_size).astype(
            "float32"
        )
        pd_x_bf16 = paddle.to_tensor(np_x).to(paddle.bfloat16)
        assert is_fp8_tensor(pd_x_bf16) is False

        fp8_x = paddle.incubate.nn.functional.fp8_quant_blockwise(
            pd_x_bf16,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
        )
        assert is_fp8_tensor(fp8_x) is True

    def test_forward_backward(self):
        np.random.seed(123)
        batch_size = 16384

        for i in range(self.acc_step):
            np_x = np.random.randn(batch_size, self.config.hidden_size).astype(
                "float32"
            )
            pd_x_fp32 = paddle.to_tensor(np_x)
            pd_x_bf16 = paddle.to_tensor(np_x).to(paddle.bfloat16)

            pd_x_fp32.stop_gradient = False
            pd_x_bf16.stop_gradient = False

            out_fp32, _ = self.fp32_linear(pd_x_fp32)
            out_fp32.sum().backward()

            out_fp8, _ = self.fp8_linear(pd_x_bf16)
            out_fp8.sum().backward()

            out_diff = calc_diff(out_fp32, out_fp8)
            assert out_diff < 0.001, f"iter {i} failed, out_diff: {out_diff}"

            w_grad_diff = calc_diff(
                self.fp32_linear.weight.grad, self.fp8_linear.weight.main_grad.T
            )
            x_grad_diff = calc_diff(pd_x_fp32.grad, pd_x_bf16.grad)
            assert w_grad_diff < 0.001, (
                f"iter {i} failed, w_grad_diff: {w_grad_diff}"
            )
            assert x_grad_diff < 0.001, (
                f"iter {i} failed, x_grad_diff: {x_grad_diff}"
            )

    def test_transformer_layer(self):
        vocab_size = 12800
        seq_len = 4096
        batch_size = 2
        fp32_config = GPTConfig(
            vocab_size=vocab_size,
            max_sequence_length=seq_len,
            num_hidden_layers=2,
            hidden_size=self.config.hidden_size,
            num_attention_heads=4,
            intermediate_size=self.config.intermediate_size,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            parallel_output=True,
            tie_word_embeddings=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            use_qk_norm=True,
        )

        fp8_config = copy.deepcopy(fp32_config)
        fp8_config.fp8 = "e4m3"
        fp8_config.fp8_linear = True

        set_seed(46)
        fp32_gpt_model = gpt_builder(fp32_config, num_stages=1)
        set_seed(46)
        fp8_gpt_model = gpt_builder(fp8_config, num_stages=1)
        paddle.amp.decorate(
            models=fp8_gpt_model,
            level="O2",
            dtype="bfloat16",
            master_grad=True,
        )
        mix_precision_utils.MixPrecisionLayer(fp8_gpt_model, "bfloat16")

        strategy = fleet.DistributedStrategy()
        fp32_gpt_model = NoPipelineParallel(fp32_gpt_model, strategy)
        fp8_gpt_model = NoPipelineParallel(fp8_gpt_model, strategy)

        for i in range(self.acc_step):
            data = paddle.randint(
                low=0, high=vocab_size, shape=(batch_size, seq_len + 1)
            )

            input_ids = data[:, :-1]
            labels = data[:, 1:]
            position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
                (batch_size, 1)
            )
            inputs = (
                {
                    "input_ids": [input_ids],
                    "position_ids": [position_ids],
                },
                [labels],
            )
            fp32_loss = fp32_gpt_model.forward_backward_pipeline(inputs)
            fp8_loss = fp8_gpt_model.forward_backward_pipeline(inputs)

            assert fp32_loss - fp8_loss < 1e-3, (
                f"iter {i} failed, fp32_loss: {fp32_loss}, fp8_loss: {fp8_loss}"
            )


if __name__ == "__main__":
    unittest.main()
