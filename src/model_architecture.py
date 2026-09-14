"""
src/model_architecture.py

Deep learning architecture for Hiligaynon cognitive fatigue detection.
Contains serializable Temporal Attention mechanism and dual Conv2D backbone
matching the thesis specifications and Keras 3 serialization standards.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

# Cross-compatible serialization decorator import
try:
    from tensorflow.keras.utils import register_keras_serializable
except (ImportError, AttributeError):
    import keras
    register_keras_serializable = keras.saving.register_keras_serializable


@register_keras_serializable(package="CustomLayers")
class TemporalAttention(layers.Layer):
    """
    Temporal Attention mechanism applying tanh projection and Softmax normalization
    across the time dimension to highlight acoustic fatigue markers.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape: Tuple[int, ...]) -> None:
        feature_dim = input_shape[-1]
        self.w = self.add_weight(
            name="attention_w",
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.v = self.add_weight(
            name="context_v",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # inputs shape: (batch_size, time_steps, feature_dim)
        score = tf.nn.tanh(tf.matmul(inputs, self.w) + self.b)
        energy = tf.matmul(score, self.v)
        weights = tf.nn.softmax(energy, axis=1)

        # Weighted context pooling across temporal axis
        context_vector = tf.reduce_sum(inputs * weights, axis=1)
        weights_squeezed = tf.squeeze(weights, axis=-1)

        return context_vector, weights_squeezed

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_fatigue_model(
    input_shape: Tuple[int, int, int] = (128, 1500, 1),
    num_classes: int = 3,
    l2_reg: float = 1e-4,
    return_attention: bool = False,
) -> Model:
    """
    Builds the Conv2D + Temporal Attention architecture.

    Parameters:
        input_shape: Dimensions of normalized Mel-spectrogram (n_mels, frames, channels).
        num_classes: Classification output nodes (Low, Moderate, High).
        l2_reg: Weight decay factor for convolutional and dense kernels.
        return_attention: When True, model returns [class_probs, attention_weights].
                          When False, returns class_probs for standard training.

    Returns:
        tf.keras.Model: Configured Keras computational graph.
    """
    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")

    # --- Convolutional Block 1 ---
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv2d_block1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_block1")(x)
    x = layers.Activation("relu", name="relu_block1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_block1")(x)
    x = layers.Dropout(0.2, name="dropout_block1")(x)

    # --- Convolutional Block 2 ---
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv2d_block2",
    )(x)
    x = layers.BatchNormalization(name="bn_block2")(x)
    x = layers.Activation("relu", name="relu_block2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_block2")(x)
    x = layers.Dropout(0.2, name="dropout_block2")(x)

    # --- Feature Alignment & Reshape for Temporal Sequence ---
    # Shape transition: (batch, freq, time, channels) -> (batch, time, freq, channels)
    x = layers.Permute((2, 1, 3), name="permute_time_first")(x)
    time_steps = x.shape[1]
    feature_dim = x.shape[2] * x.shape[3]
    x = layers.Reshape((time_steps, feature_dim), name="flatten_spatial")(x)

    # --- Temporal Attention Layer ---
    context_vector, attention_weights = TemporalAttention(
        units=64, name="temporal_attention"
    )(x)

    # --- Classification Head ---
    d = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_projection",
    )(context_vector)
    d = layers.Dropout(0.3, name="dropout_dense")(d)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="fatigue_output"
    )(d)

    if return_attention:
        return Model(
            inputs=inputs,
            outputs=[outputs, attention_weights],
            name="fatigue_cnn_attention_inference",
        )
    return Model(inputs=inputs, outputs=outputs, name="fatigue_cnn_attention_training")


def get_inference_model(trained_model: Model) -> Model:
    """
    Extracts a dual-output inference model from a trained single-output model
    without modifying learned layer weights.
    """
    attention_layer = trained_model.get_layer("temporal_attention")
    _, attention_weights = attention_layer.output
    return Model(
        inputs=trained_model.input,
        outputs=[trained_model.get_layer("fatigue_output").output, attention_weights],
        name="fatigue_inference_model",
    )