def train_model(model, X_train, Y_train, X_val, Y_val, scheduler, batch_size=64, epochs=150):
    return model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[scheduler]
    )