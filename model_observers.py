import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import telebot

import io
import warnings
import threading


'''
Simple hand-written library for telegram notifications about model fitting process.

'''




# GLOBAL TODO: make more beautiful message wrappers in feature versions,
# implement quick visualization DURING the learning process



# KERAS CALLBACK FOR SUPERVISED MODELS.
# It may be dangerous to use this type of callback on unsupervised
# or even RL models, so needs to be tested as well.

class ObserverCallback(keras.callbacks.Callback):
    def __init__(self, token, user_id, precision=7):
        super().__init__()
        self.token = token
        self.user_id = user_id
        self.bot = telebot.TeleBot(token)
        self.precision = precision
        self.history = {}


    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)

        message = f"Epoch {epoch + 1}/{self.params.get('epochs', '?')}\n"
        message += "```\n"

        for key, value in logs.items():
            try:
                if isinstance(value, (int, float, np.floating, np.integer)):
                    value = np.round(value, self.precision)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    value = ", ".join([str(np.round(v, self.precision)) for v in value])
                else:
                    value = str(value)
                message += f"{key:<15}: {value}\n"
            except Exception as e:
                warnings.warn(f"Could not format log '{key}': {e}", RuntimeWarning)

        message += "```"

        self.send_logs(message)


    def on_train_begin(self, logs=None):
        message = "Beginning training process..."
        self.send_logs(message)


    def on_train_end(self, logs=None):
        message = "Training finished."
        self.send_history()
        self.send_logs(message)


    def send_logs(self, message=None):
        if not message:
            warnings.warn("Empty log message.", UserWarning)
        self.bot.send_message(self.user_id, message, parse_mode='Markdown')


    def send_history(self):
        if not self.history:
            return
        
        plt.figure(figsize=(8, 5))
        for key, values in self.history.items():
            plt.plot(range(1, len(values) + 1), values, label=key)
        plt.legend()
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Metric values")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        try:
            self.bot.send_photo(self.user_id, buf)
        except Exception as e:
            warnings.warn(f"Failed to send training plot: {e}", RuntimeWarning)


# Sklearn callback (use sklearn_genetic.callback)
# https://sklearn-genetic-opt.readthedocs.io/en/stable/tutorials/custom_callback.html




# TODO pytorch callback 