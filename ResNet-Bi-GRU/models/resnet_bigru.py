from tensorflow.keras import layers, models
from .residual_block import residual_block

def build_resnet_bigru(input_shape, output_dim):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    for filters in [32, 32, 64, 64, 128, 128]:
        x = residual_block(x, filters)
        if filters in [32, 64]:
            x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128))(x)

    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dense(100, activation='relu')(x)
    outputs = layers.Dense(output_dim, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)