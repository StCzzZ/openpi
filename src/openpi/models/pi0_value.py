import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def _make_value_action_block_mask(
    prefix_mask: at.Bool[at.Array, "b p"],
    value_mask: at.Bool[at.Array, "b v"],
    action_mask: at.Bool[at.Array, "b s"],
) -> at.Bool[at.Array, "b _t _t"]:
    """Builds a mask that forbids value/action tokens from attending to each other."""
    prefix_ids = jnp.zeros_like(prefix_mask, dtype=jnp.int32)
    value_ids = jnp.ones_like(value_mask, dtype=jnp.int32)
    action_ids = jnp.full_like(action_mask, 2, dtype=jnp.int32)
    block_ids = jnp.concatenate([prefix_ids, value_ids, action_ids], axis=1)
    value_queries = block_ids[:, :, None] == 1
    action_queries = block_ids[:, :, None] == 2
    value_keys = block_ids[:, None, :] == 1
    action_keys = block_ids[:, None, :] == 2
    value_to_action_mask = jnp.logical_not(value_queries & action_keys)
    action_to_value_mask = jnp.logical_not(action_queries & value_keys)
    return jnp.logical_and(value_to_action_mask, action_to_value_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0SuffixValue(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0SuffixValueConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.discrete_value = config.discrete_value
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        if config.value_variant != config.action_expert_variant:
            raise ValueError("Pi0SuffixValue requires `value_variant` to match `action_expert_variant` when sharing experts.")
        value_config = action_expert_config
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Learnable parameters for value head input
        self.value_params = nnx.Param(jnp.zeros((1, value_config.width)))
        if self.discrete_value:
            self.value_out_proj = nnx.Linear(value_config.width, config.n_bins, rngs=rngs)
        else:
            self.value_out_proj = nnx.Linear(value_config.width, 1, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_value_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tokens = self.value_params[:, None, :].repeat(obs.state.shape[0], axis=0)
        input_mask = jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_)
        ar_mask = jnp.array([True])
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        if self.pi05:
            adarms_cond = self._compute_adarms_cond(timestep)
            action_expert_tokens = action_tokens
        else:
            # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _compute_adarms_cond(
        self, timestep: at.Float[at.Array, " b"]
    ) -> at.Float[at.Array, "b emb"] | None:
        """Computes the adaRMS conditioning vector for the action expert when pi05 is enabled."""
        if not self.pi05:
            return None
        timestep = timestep.astype(jnp.float32)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        return time_emb

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, value: _model.Values, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + value + suffix at once (value and action experts share weights)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        value_tokens, value_mask, value_ar_mask = self.embed_value_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        expert_tokens = jnp.concatenate([value_tokens, suffix_tokens], axis=1)
        expert_mask = jnp.concatenate([value_mask, suffix_mask], axis=1)
        expert_ar_mask = jnp.concatenate([value_ar_mask, suffix_ar_mask], axis=0)
        input_mask = jnp.concatenate([prefix_mask, expert_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, expert_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        attn_mask = jnp.logical_and(attn_mask, _make_value_action_block_mask(prefix_mask, value_mask, suffix_mask))
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, expert_out), _ = self.PaliGemma.llm(
            [prefix_tokens, expert_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        value_out, suffix_out = jnp.split(expert_out, [value_tokens.shape[1]], axis=1)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        value_pred = self.value_out_proj(value_out)
        if self.discrete_value:
            value_pred = nnx.log_softmax(value_pred, axis=-1)
            value_loss = 0.1 * jnp.sum(-value[:, None, :] * value_pred, axis=-1)
        else:
            value_pred = nnx.sigmoid(value_pred)
            value_loss = jnp.mean(jnp.square(value_pred + value[:, None, :]), axis=-1) # because the data has negative values
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        return action_loss + value_loss, action_loss, value_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, _model.Values]:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_value_time = jnp.ones((batch_size,), dtype=jnp.float32)
        prefix_value_adarms_cond = self._compute_adarms_cond(prefix_value_time)

        # first fill KV cache with a forward pass of the prefix and value tokens
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        value_tokens, value_mask, value_ar_mask = self.embed_value_prefix(observation)
        prefix_value_mask = jnp.concatenate([prefix_mask, value_mask], axis=1)
        prefix_value_ar_mask = jnp.concatenate([prefix_ar_mask, value_ar_mask], axis=0)
        prefix_value_attn_mask = make_attn_mask(prefix_value_mask, prefix_value_ar_mask)
        positions = jnp.cumsum(prefix_value_mask, axis=1) - 1
        (_, value_pred_out), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, value_tokens],
            mask=prefix_value_attn_mask,
            positions=positions,
            adarms_cond=[None, prefix_value_adarms_cond],
        )
        value_pred = self.value_out_proj(value_pred_out)
        if self.discrete_value:
            value_pred = nnx.softmax(value_pred, axis=-1)
            value_pred = value_pred[:, -1, :]
        else:
            value_pred = nnx.sigmoid(value_pred)
            value_pred = -value_pred[:, -1, :] # because the data should have negative values

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            value_attn_mask = einops.repeat(jnp.full_like(value_mask, False), "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + value_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, value_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + value_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_value_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, expert_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None and expert_out is not None
            v_t = self.action_out_proj(expert_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, value_pred



class Pi0PrefixValue(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0PrefixValueConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.discrete_value = config.discrete_value
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        if config.value_variant != config.paligemma_variant:
            raise ValueError("Pi0PrefixValue requires `value_variant` to match `paligemma_variant` when sharing experts.")
        value_config = paligemma_config
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Learnable parameters for value head input
        self.value_params = nnx.Param(jnp.zeros((1, value_config.width)))
        if self.discrete_value:
            self.value_out_proj = nnx.Linear(value_config.width, config.n_bins, rngs=rngs)
        else:
            self.value_out_proj = nnx.Linear(value_config.width, 1, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_value_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tokens = self.value_params[:, None, :].repeat(obs.state.shape[0], axis=0)
        input_mask = jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_)
        ar_mask = jnp.array([True])
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        if self.pi05:
            adarms_cond = self._compute_adarms_cond(timestep)
            action_expert_tokens = action_tokens
        else:
            # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _compute_adarms_cond(
        self, timestep: at.Float[at.Array, " b"]
    ) -> at.Float[at.Array, "b emb"] | None:
        """Computes the adaRMS conditioning vector for the action expert when pi05 is enabled."""
        if not self.pi05:
            return None
        timestep = timestep.astype(jnp.float32)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        return time_emb

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, value: _model.Values, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + value + suffix at once (prefix and value experts share weights)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        value_tokens, value_mask, value_ar_mask = self.embed_value_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        total_prefix_tokens = jnp.concatenate([prefix_tokens, value_tokens], axis=1)
        total_prefix_mask = jnp.concatenate([prefix_mask, value_mask], axis=1)
        total_prefix_ar_mask = jnp.concatenate([prefix_ar_mask, value_ar_mask], axis=0)
        input_mask = jnp.concatenate([total_prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([total_prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # attn_mask = jnp.logical_and(attn_mask, _make_value_action_block_mask(prefix_mask, value_mask, suffix_mask))
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, expert_out), _ = self.PaliGemma.llm(
            [total_prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        _, value_out = jnp.split(prefix_out, [prefix_tokens.shape[1]], axis=1)
        v_t = self.action_out_proj(expert_out[:, -self.action_horizon :])
        value_pred = self.value_out_proj(value_out)
        if self.discrete_value:
            value_pred = nnx.log_softmax(value_pred, axis=-1)
            value_loss = 0.1 * jnp.sum(-value[:, None, :] * value_pred, axis=-1)
        else:
            value_pred = nnx.sigmoid(value_pred)
            value_loss = jnp.mean(jnp.square(value_pred + value[:, None, :]), axis=-1) # because the data has negative values
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        return action_loss + value_loss, action_loss, value_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, _model.Values]:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix and value tokens
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        value_tokens, value_mask, value_ar_mask = self.embed_value_prefix(observation)
        total_prefix_tokens = jnp.concatenate([prefix_tokens, value_tokens], axis=1)
        total_prefix_mask = jnp.concatenate([prefix_mask, value_mask], axis=1)
        total_prefix_ar_mask = jnp.concatenate([prefix_ar_mask, value_ar_mask], axis=0)
        total_prefix_attn_mask = make_attn_mask(total_prefix_mask, total_prefix_ar_mask)
        positions = jnp.cumsum(total_prefix_mask, axis=1) - 1
        (total_prefix_out, _), kv_cache = self.PaliGemma.llm(
            [total_prefix_tokens, None],
            mask=total_prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
        )
        _, value_out = jnp.split(total_prefix_out, [prefix_tokens.shape[1]], axis=1)
        value_pred = self.value_out_proj(value_out)
        if self.discrete_value:
            value_pred = nnx.softmax(value_pred, axis=-1)
            value_pred = value_pred[:, -1, :]
        else:
            value_pred = nnx.sigmoid(value_pred)
            value_pred = -value_pred[:, -1, :] # because the data should have negative values

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            total_prefix_attn_mask = einops.repeat(total_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # value_attn_mask = einops.repeat(jnp.full_like(value_mask, False), "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + value_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([total_prefix_attn_mask, suffix_attn_mask], axis=-1)
            # full_attn_mask = jnp.concatenate([prefix_attn_mask, value_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + value_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(total_prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, expert_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None and expert_out is not None
            v_t = self.action_out_proj(expert_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, value_pred