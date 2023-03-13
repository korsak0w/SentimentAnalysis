######################
# Import libraries
######################
import tensorflow as tf
import streamlit as st
import time


class PrintCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_taken = time.time() - self.start_time
        st.text(f"Epoch {epoch+1}/{self.epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - time: {time_taken:.2f}s")


class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, steps):
        self.steps = steps
        self.progress_bar = st.progress(0)
    
    def on_train_batch_end(self, batch, logs=None):
        progress_value = 100 / self.steps
        self.progress_bar.progress(int(progress_value * batch))

