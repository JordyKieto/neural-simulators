"""TinySkies neural simulator: a frozen spatial VAE plus a recurrent latent transition model.

    z_t = E(o_t)                       o_t: RGB 96x96x3 in [0,1],  z_t: 6x6x64 (2304-d)
    z_{t+1} = z_t + g(z_{t-3:t}, a_{t-4:t-1}, a_t)
    o_{t+1} = D(z_{t+1})

a_t = (forward, brake, turn) from the W / S / A,D keys. The simulator never sees pixels after the
seed: it rolls forward entirely in latent space, and the decoder is only a viewer.

Parameters: encoder 660,480 + decoder 1,282,933 (the VAE's log-variance head is training-only
and not exported) + latent simulator 1,842,071 = 3,785,484 at runtime.
"""

import keras
from keras import layers

VISION_SHAPE = (96, 96, 3)
LATENT_SHAPE = (6, 6, 64)
HISTORY = 4
CONTROLS_DIM = 3
CONTROL_NAMES = ("forward", "brake", "turn")


def build_encoder():
    """RGB -> z_mean (the runtime encoder; the VAE's z_log_var head is only used in training)."""
    rgb = keras.Input(VISION_SHAPE, name="rgb")
    x = layers.Conv2D(32, 5, strides=2, padding="same", activation="swish")(rgb)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="swish")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="swish")(x)
    x = layers.Conv2D(192, 3, strides=2, padding="same", activation="swish")(x)
    x = layers.Conv2D(192, 3, padding="same", activation="swish")(x)
    z_mean = layers.Conv2D(LATENT_SHAPE[-1], 1, padding="same", name="z_mean")(x)
    return keras.Model(rgb, z_mean, name="tinyskies_encoder")


def build_decoder():
    z = keras.Input(LATENT_SHAPE, name="z")
    x = layers.Conv2D(192, 3, padding="same", activation="swish")(z)
    for filters in (192, 128, 64, 32):
        x = layers.UpSampling2D(2, interpolation="nearest")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="swish")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="swish")(x)
    reconstruction = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="reconstruction")(x)
    return keras.Model(z, reconstruction, name="tinyskies_decoder")


@keras.saving.register_keras_serializable(package="TinySkiesVAE")
class Sampling(layers.Layer):
    """Reparameterization z = mu + sigma * eps. Training only; runtime uses z_mean."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = keras.random.normal(keras.ops.shape(z_mean), dtype=z_mean.dtype)
        return z_mean + keras.ops.exp(0.5 * z_log_var) * eps

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def load_checkpoints(vae_path, sim_path):
    """Load the released Keras checkpoints -> (encoder, decoder, latent_sim) runtime models."""
    vae = keras.models.load_model(vae_path, compile=False, safe_mode=False)
    enc = vae.get_layer("tinyskies_rgb_vae_encoder")
    encoder = keras.Model(enc.input, enc.output[0], name="tinyskies_encoder")
    decoder = vae.get_layer("tinyskies_rgb_vae_decoder")
    sim = keras.models.load_model(sim_path, compile=False, safe_mode=False)
    return encoder, decoder, sim


@keras.saving.register_keras_serializable(package="TinySkiesLatentSim")
class TakeCurrentLatent(layers.Layer):
    def call(self, inputs):
        return inputs[:, -1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + tuple(input_shape[2:])


def build_latent_sim(model_dim=128, rnn_units=256, ff_dim=256, dropout=0.05):
    """(latent_history 4x6x6x64, control_history 4x3, controls 3) -> next_latent, latent_delta.

    control_history[j] is the action that led *into* latent_history[j] (zeros for the first);
    `controls` is the action applied at the newest latent.
    """
    tokens = LATENT_SHAPE[0] * LATENT_SHAPE[1] * LATENT_SHAPE[2]
    latent_history = keras.Input((HISTORY,) + LATENT_SHAPE, name="latent_history")
    control_history = keras.Input((HISTORY, CONTROLS_DIM), name="control_history")
    controls = keras.Input((CONTROLS_DIM,), name="controls")

    latent_steps = layers.Reshape((HISTORY, tokens), name="latent_history_flat")(latent_history)
    latent_steps = layers.Dense(model_dim, activation="swish", name="latent_step_embed")(latent_steps)
    control_steps = layers.Dense(model_dim, activation="swish", name="control_history_embed")(control_history)
    x = layers.Concatenate(axis=-1, name="rnn_sequence_inputs")([latent_steps, control_steps])
    x = layers.GRU(rnn_units, return_sequences=True, dropout=dropout, name="latent_dynamics_gru_1")(x)
    x = layers.GRU(rnn_units, dropout=dropout, name="latent_dynamics_gru_2")(x)

    current = layers.Dense(model_dim, activation="swish", name="current_control_embed")(controls)
    h = layers.Concatenate(name="transition_context")([x, current])
    h = layers.Dense(ff_dim, activation="swish", name="transition_head_dense_1")(h)
    h = layers.Dropout(dropout, name="transition_head_dropout")(h)
    h = layers.Dense(ff_dim, activation="swish", name="transition_head_dense_2")(h)
    delta = layers.Reshape(LATENT_SHAPE, name="latent_delta")(layers.Dense(tokens, name="latent_delta_flat")(h))

    # Residual: the model predicts only the change from the newest latent.
    next_latent = layers.Add(name="next_latent")([TakeCurrentLatent(name="current_latent")(latent_history), delta])
    return keras.Model(
        [latent_history, control_history, controls],
        {"next_latent": next_latent, "latent_delta": delta},
        name="tinyskies_latent_sim_v1",
    )
