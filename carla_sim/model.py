"""CARLA neural simulator (world_v3): a one-step, action-conditioned multimodal transition model.

State s_t = (RGB 96x96x3 in [0,1], LiDAR latent 64, telemetry 5, waypoints 25x3).
Action a_t = (throttle, brake, steer).

    s_{t+1} = f(s_t, a_t)

The model is Markov in s_t: no frame history. Anything that must persist over time (speed,
heading, momentum) has to be carried in the predicted state itself. LiDAR is encoded once, from
the seed frame's bird's-eye-view histogram, and from then on it evolves only as a predicted
64-d latent. There is no LiDAR decoder.

Parameters: transition 1,551,854 + LiDAR encoder 27,462 = 1,579,316.
"""

import numpy as np
import keras
from keras import layers

VISION_SHAPE = (96, 96, 3)
LIDAR_SHAPE = (64, 64, 1)
LIDAR_LATENT_DIM = 64
TELEMETRY_DIM = 5
WAYPOINTS_SHAPE = (25, 3)
CONTROLS_DIM = 3

TELEMETRY_NAMES = ("speed_mps", "speed_kph", "accel_mps2", "v_forward", "v_lateral")
CONTROL_NAMES = ("throttle", "brake", "steer")


def build_lidar_encoder():
    """LiDAR BEV histogram (64x64x1) -> 64-d latent."""
    return keras.Sequential(
        [
            keras.Input(LIDAR_SHAPE, name="lidar"),
            layers.Conv2D(16, 3, strides=2, padding="same", activation="swish"),
            layers.Conv2D(32, 3, strides=2, padding="same", activation="swish"),
            layers.Conv2D(64, 3, strides=2, padding="same", activation="swish"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(LIDAR_LATENT_DIM, activation="swish"),
        ],
        name="lidar_encoder",
    )


def build_transition():
    """(vision, lidar_latent, telemetry, waypoints, controls) -> next state (same four fields)."""
    vision = keras.Input(VISION_SHAPE, name="vision")
    lidar_latent = keras.Input((LIDAR_LATENT_DIM,), name="lidar_latent")
    telemetry = keras.Input((TELEMETRY_DIM,), name="telemetry")
    waypoints = keras.Input(WAYPOINTS_SHAPE, name="waypoints")
    controls = keras.Input((CONTROLS_DIM,), name="controls")

    # Image encoder: 96 -> 6 spatial, 192 channels.
    x = layers.Conv2D(32, 5, strides=2, padding="same", activation="swish")(vision)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="swish")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="swish")(x)
    x = layers.Conv2D(192, 3, strides=2, padding="same", activation="swish")(x)

    # Structured state and action, fused into one vector.
    state_action = layers.Concatenate()(
        [
            layers.Dense(64, activation="swish")(lidar_latent),
            layers.Dense(32, activation="swish")(telemetry),
            layers.Dense(64, activation="swish")(layers.Flatten()(waypoints)),
            layers.Dense(64, activation="swish")(controls),
        ]
    )

    # The action reaches the image path only through this broadcast 6x6x48 conditioning map.
    condition = layers.Reshape((6, 6, 48))(layers.Dense(6 * 6 * 48, activation="swish")(state_action))
    x = layers.Concatenate()([x, condition])

    # Image decoder: 6 -> 96. Predicts the next frame directly (no residual).
    x = layers.Conv2D(192, 3, padding="same", activation="swish")(x)
    for channels in (128, 64, 32):
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(channels, 3, padding="same", activation="swish")(x)
    x = layers.UpSampling2D()(x)
    next_vision = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="next_vision")(x)

    # Structured-state head.
    head = layers.Dense(192, activation="swish")(state_action)
    head = layers.Dense(192, activation="swish")(head)
    next_lidar_latent = layers.Dense(LIDAR_LATENT_DIM, name="next_lidar_latent")(head)
    next_telemetry = layers.Dense(TELEMETRY_DIM, name="next_telemetry")(head)
    next_waypoints = layers.Reshape(WAYPOINTS_SHAPE, name="next_waypoints")(
        layers.Dense(int(np.prod(WAYPOINTS_SHAPE)), name="next_waypoints_flat")(head)
    )

    return keras.Model(
        [vision, lidar_latent, telemetry, waypoints, controls],
        {
            "next_vision": next_vision,
            "next_lidar_latent": next_lidar_latent,
            "next_telemetry": next_telemetry,
            "next_waypoints": next_waypoints,
        },
        name="world_v3_transition",
    )


def speed_mps(telemetry):
    """Invert the telemetry soft-normalization: channel 0 is tanh(speed / 20 m/s)."""
    t = np.clip(np.asarray(telemetry)[..., 0], -0.999999, 0.999999)
    return 20.0 * np.arctanh(t)
