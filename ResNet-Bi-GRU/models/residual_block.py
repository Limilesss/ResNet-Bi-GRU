from tensorflow.keras import layers

def residual_block(x, filters, kernel_size=3, stride=1):
    res = x
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if res.shape[-1] != filters:
        res = layers.Conv1D(filters, kernel_size=1, padding='same')(res)

    x = layers.Add()([x, res])
    x = layers.Activation('relu')(x)
    return x