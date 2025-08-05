from tensorflow.keras import layers, models
from .residual_block import residual_block

def build_resnet_gru_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = residual_block(x, filters=32)
    x = residual_block(x, filters=32)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128))(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dense(100, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
