from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

try:
    from .configuration_rwkv7 import RWKV7Config
except ImportError:  # pragma: no cover - direct local execution
    from configuration_rwkv7 import RWKV7Config


LOG_DECAY_SCALE = -0.6065306597126334


@dataclass
class RWKV7Output(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    state: Optional[list[torch.Tensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class RWKV7CausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor | None = None
    state: Optional[list[torch.Tensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


def sqrelu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


def get_activation_fn(name: str):
    if name == "sqrelu":
        return sqrelu
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported RWKV7 activation: {name}")


def token_shift(hidden_states: torch.Tensor, cached_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.empty_like(hidden_states)
    if cached_state is None:
        delta[:, 0] = -hidden_states[:, 0]
    else:
        delta[:, 0] = cached_state.to(hidden_states.dtype) - hidden_states[:, 0]
    if hidden_states.shape[1] > 1:
        delta[:, 1:] = hidden_states[:, :-1] - hidden_states[:, 1:]
    return delta, hidden_states[:, -1]


class RWKV7LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool,
        activation: str | None,
    ) -> None:
        super().__init__()
        if activation is None:
            act = nn.Identity()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported RWKV7 LoRA activation: {activation}")
        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            act,
            nn.Linear(low_rank_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class RWKV7GroupNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, value_dim: int, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.eps = head_dim * eps
        self.weight = nn.Parameter(torch.ones(value_dim))
        self.bias = nn.Parameter(torch.zeros(value_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = F.group_norm(
            x.float().unsqueeze(-1),
            num_groups=self.num_heads,
            weight=self.weight.float(),
            bias=self.bias.float(),
            eps=self.eps,
        )
        return x.squeeze(-1).to(dtype)


class RWKV7FeedForward(nn.Module):
    def __init__(self, config: RWKV7Config, layer_idx: int) -> None:
        super().__init__()
        del layer_idx
        if config.intermediate_size is None:
            hidden_ratio = 4 if config.hidden_ratio is None else config.hidden_ratio
            intermediate_size = int(config.hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 31) // 32)
        else:
            intermediate_size = config.intermediate_size

        self.x_k = nn.Parameter(torch.zeros(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = get_activation_fn(config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cached_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta, final_state = token_shift(hidden_states, cached_state)
        mixed = hidden_states + delta * self.x_k
        return self.value(self.act_fn(self.key(mixed))), final_state


class RWKV7Attention(nn.Module):
    def __init__(self, config: RWKV7Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.value_dim = config.value_dim[layer_idx]
        self.head_v_dim = self.value_dim // self.num_heads

        self.x_r = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.hidden_size))
        self.k_a = nn.Parameter(torch.zeros(self.hidden_size))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.w_lora = RWKV7LoRA(
            self.hidden_size,
            self.hidden_size,
            config.decay_low_rank_dim,
            bias=True,
            activation="tanh",
        )
        self.a_lora = RWKV7LoRA(
            self.hidden_size,
            self.hidden_size,
            config.a_low_rank_dim,
            bias=True,
            activation=None,
        )
        if self.layer_idx != 0:
            self.v_lora = RWKV7LoRA(
                self.hidden_size,
                self.value_dim,
                config.v_low_rank_dim,
                bias=True,
                activation=None,
            )
        self.g_lora = RWKV7LoRA(
            self.hidden_size,
            self.value_dim,
            config.gate_low_rank_dim,
            bias=False,
            activation="sigmoid",
        )
        self.g_norm = RWKV7GroupNorm(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            value_dim=self.value_dim,
            eps=config.norm_eps,
        )

    def _project(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xr = hidden_states + delta * self.x_r
        xw = hidden_states + delta * self.x_w
        xk = hidden_states + delta * self.x_k
        xv = hidden_states + delta * self.x_v
        xa = hidden_states + delta * self.x_a
        xg = hidden_states + delta * self.x_g

        r = self.r_proj(xr)
        w = LOG_DECAY_SCALE * self.w_lora(xw).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        if self.layer_idx == 0:
            v_first_out = v
        else:
            if v_first is None:
                raise ValueError("RWKV7 layers after layer 0 require `v_first`.")
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
            v_first_out = v_first
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        batch_size, seq_len, _ = hidden_states.shape
        r = r.view(batch_size, seq_len, self.num_heads, self.head_dim).float()
        w = w.view(batch_size, seq_len, self.num_heads, self.head_dim).float()
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).float()
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim).float()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_v_dim).float()

        k_k = self.k_k.view(self.num_heads, self.head_dim).float()
        k_a = self.k_a.view(self.num_heads, self.head_dim).float()
        kk = F.normalize(k * k_k, dim=-1, p=2.0)
        k = k * (1 + (a - 1) * k_a)
        return r, w, k, v, kk, a, g, v_first_out

    def _run_recurrent(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for idx in range(r.shape[1]):
            sa = (recurrent_state * (-kk[:, idx]).unsqueeze(-1)).sum(dim=-2)
            recurrent_state = (
                torch.exp(w[:, idx]).unsqueeze(-1) * recurrent_state
                + (kk[:, idx] * a[:, idx]).unsqueeze(-1) * sa.unsqueeze(-2)
                + k[:, idx].unsqueeze(-1) * v[:, idx].unsqueeze(-2)
            )
            outputs.append((recurrent_state * r[:, idx].unsqueeze(-1)).sum(dim=-2))
        return torch.stack(outputs, dim=1), recurrent_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        cached_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, final_shift_state = token_shift(hidden_states, cached_shift_state)
        r, w, k, v, kk, a, g, v_first_out = self._project(hidden_states, delta, v_first)

        if recurrent_state is None:
            recurrent_state = torch.zeros(
                hidden_states.shape[0],
                self.num_heads,
                self.head_dim,
                self.head_v_dim,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        else:
            recurrent_state = recurrent_state.float()

        recurrent_output, final_recurrent_state = self._run_recurrent(
            r, w, k, v, kk, a, recurrent_state
        )
        output = recurrent_output.reshape(hidden_states.shape[0], hidden_states.shape[1], self.value_dim)
        output = self.g_norm(output.reshape(-1, self.value_dim)).view_as(output)
        correction = (
            (r * k * self.r_k.float().view(1, 1, self.num_heads, self.head_dim)).sum(dim=-1, keepdim=True) * v
        ).reshape_as(output)
        output = (output.float() + correction) * g.float()
        output = self.o_proj(output.to(hidden_states.dtype))
        return output, final_shift_state, final_recurrent_state, v_first_out


class RWKV7Block(nn.Module):
    def __init__(self, config: RWKV7Config, layer_idx: int) -> None:
        super().__init__()
        self.pre_norm = None
        if config.norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                elementwise_affine=True,
                bias=config.norm_bias,
            )
        self.attn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.attn = RWKV7Attention(config, layer_idx)
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.ffn = RWKV7FeedForward(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: list[torch.Tensor],
        layer_idx: int,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        residual = self.pre_norm(hidden_states) if self.pre_norm is not None else hidden_states
        attn_out, attn_shift, recurrent, v_first = self.attn(
            self.attn_norm(residual),
            state[0][layer_idx],
            state[1][layer_idx],
            v_first,
        )
        hidden_states = residual + attn_out
        ffn_out, ffn_shift = self.ffn(self.ffn_norm(hidden_states), state[2][layer_idx])
        hidden_states = hidden_states + ffn_out
        state[0][layer_idx] = attn_shift
        state[1][layer_idx] = recurrent
        state[2][layer_idx] = ffn_shift
        return hidden_states, state, v_first


class RWKV7PreTrainedModel(PreTrainedModel):
    config_class = RWKV7Config
    base_model_prefix = "model"
    _no_split_modules = ["RWKV7Block"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        return


class RWKV7Model(RWKV7PreTrainedModel):
    def __init__(self, config: RWKV7Config) -> None:
        super().__init__(config)
        if config.attn is not None:
            raise NotImplementedError("Hybrid RWKV7 checkpoints are not supported by this PyTorch loader yet.")
        if len(set(config.value_dim)) != 1:
            raise NotImplementedError("Per-layer `value_dim` variation is not supported by this PyTorch loader yet.")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [RWKV7Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def _init_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
        value_dim = self.config.value_dim[0]
        return [
            torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            ),
            torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                self.config.num_heads,
                self.config.head_dim,
                value_dim // self.config.num_heads,
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            ),
        ]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: list[torch.Tensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        del attention_mask
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if output_attentions:
            raise NotImplementedError("RWKV7 does not expose attention matrices.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds.")
            inputs_embeds = self.embed_tokens(input_ids)

        if state is None:
            state = self._init_state(inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device)

        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        v_first = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, state, v_first = layer(hidden_states, state, layer_idx, v_first)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, state, all_hidden_states, None)
        return RWKV7Output(
            last_hidden_state=hidden_states,
            state=state if (use_cache if use_cache is not None else self.config.use_cache) else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class RWKV7ForCausalLM(RWKV7PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: RWKV7Config) -> None:
        super().__init__(config)
        self.model = RWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        if state is not None:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs["state"] = state
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        model_kwargs["state"] = outputs.state
        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: list[torch.Tensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return RWKV7CausalLMOutput(
            loss=loss,
            logits=logits,
            state=outputs.state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


Rwkv7PreTrainedModel = RWKV7PreTrainedModel
Rwkv7Model = RWKV7Model
Rwkv7ForCausalLM = RWKV7ForCausalLM
