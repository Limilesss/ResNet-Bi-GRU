from tensorflow.keras.callbacks import Callback
import math
import tensorflow.keras.backend as K

class CosineAnnealingScheduler(Callback):
    def __init__(self, T_max, eta_min=0):
        super().__init__()
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = None

    def on_train_begin(self, logs=None):
        self.initial_lr = float(K.get_value(self.model.optimizer.lr))

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)