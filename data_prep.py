from sklearn.utils import shuffle
import pandas as pd 
import numpy as np


from image_procesing import get_data

IMG_SIZE = (256,256)

train_image_dir = "train_v2"
train_encode_file = "train_ship_segmentations_v2.csv"



df = pd.read_csv(train_encode_file, index_col=0).fillna('0')

df = df.groupby("ImageId")[['EncodedPixels']].agg(lambda rle_codes: ' '.join(rle_codes)).reset_index()

# marking existence ships on image
df.loc[df['EncodedPixels'] == '0', 'ship'] = 0 
df.loc[df['EncodedPixels'] != '0', 'ship'] = 1 
df.sort_values(by=['ship'])
df.value_counts('ship')

# spliting dataframe
empty = df[df['ship'] == 0]
empty = empty.reset_index()

ships = df[df['ship'] == 1]
ships = ships.reset_index()



x_clr,y_clr = get_data(train_image_dir, 0, 50, empty)
x,y = get_data(train_image_dir, 0, 2000, ships)
X_train = np.concatenate((np.array(x_clr),np.array(x)),axis=0)
Y_train = np.concatenate((np.array(y_clr),np.array(y)),axis=0)


X_train, Y_train = shuffle(X_train, Y_train, random_state=0)


x_test,y_test = get_data(train_image_dir, 2000, 2200, ships)
X_test = np.array(x)
Y_test = np.array(y)