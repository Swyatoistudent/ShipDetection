
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from model import model
from image_procesing import get_data, normalize
from display import display
from PIL import Image

test_image_dir = "test_v2"
test_encode_file = "sample_submission_v2.csv"

single_test_image = 'test_v2/00a3ab3cc.jpg'

df = pd.read_csv(test_encode_file, index_col=0).fillna('0')
df = df.groupby("ImageId")[['EncodedPixels']].agg(
    lambda rle_codes: ' '.join(rle_codes)).reset_index()
X_test, y_test = get_data(test_image_dir, 0, 100, df)


# path to model weightss
weights_dir = 'model.h5'


model.load_weights(weights_dir)


# test random sample from test_dataset
# index = random.randint(0, len(X_test))
# sample_image = X_test[index]
# sample_mask = y_test[index]
# prediction = model.predict(sample_image[tf.newaxis, ...])[0]
# predicted_mask = (prediction>0.5).astype(np.uint8)
# display([sample_image, sample_mask,predicted_mask])


# single image test
sample_image = Image.open(single_test_image)
sample_image, msk = normalize(sample_image, [])
prediction = model.predict(sample_image[tf.newaxis, ...])[0]
predicted_mask = (prediction > 0.5).astype(np.uint8)
display([sample_image, predicted_mask], ["Input image", "Predicted Mask"])
