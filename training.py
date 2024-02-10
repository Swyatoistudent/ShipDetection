from model import model
import tensorflow as tf
from tensorflow.keras import backend as K
from IPython.display import clear_output
import os

from test_model import display
from data_prep import X_train,Y_train,X_test,Y_test

def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth=10e-6):
    return 1 - dice_coef(y_true, y_pred , smooth)

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    sample_image = X_test[1]
    sample_mask = Y_test[1]
    prediction = model.predict(sample_image[tf.newaxis, ...])[0]
    prediction = (prediction>0.5).astype(np.uint8)
    display([sample_image,sample_mask,prediction])
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

    
checkpoint_dir = os.path.dirname("weights/")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        DisplayCallback(),
        cp_callback
]

model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy'])
model.fit(X_train, Y_train, validation_split= 0.2, batch_size=32, epochs=40, callbacks=callbacks)
