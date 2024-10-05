import tensorflow as tf

# Losses
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))  # MSE

# Metrics
def custom_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))  # MAE
