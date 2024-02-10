from sklearn.utils import shuffle
import pandas as pd 
import numpy as np
from PIL import Image
import skimage

IMG_SIZE = (256,256)

train_image_dir = "train_v2"
train_encode_file = "train_ship_segmentations_v2.csv"
test_image_dir= "test_v2"

def rle_to_pixels(rle_code):
    '''
    Transforms a RLE code string into a list of pixels of a (768, 768)
    '''
    if rle_code ==0:
        return [(0,0)]
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % 768, pixel_position // 768) 
                 for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 
                 for pixel_position in range(start, start + length)]
    return pixels


df = pd.read_csv(train_encode_file, index_col=0).fillna('0')

df = df.groupby("ImageId")[['EncodedPixels']].agg(lambda rle_codes: ' '.join(rle_codes)).reset_index()

load_img = lambda filename: np.array(Image.open(f"/kaggle/input/airbus-ship-detection/train_v2/{filename}"))



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

def normalize(input_image,input_mask):
    input_image = tf.cast(input_image,tf.float32) / 255.0
    return input_image, input_mask

def get_image(index,df):
    '''
    Get image and mask. Resize and normalize it
    '''
    input_image = load_img(df.loc[index, 'ImageId'])
    input_image= tf.image.resize(input_image, IMG_SIZE,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   
    mask_pixels = rle_to_pixels(df.loc[index, 'EncodedPixels'])
    canvas = np.zeros((768,768))
    canvas[tuple(zip(*mask_pixels))] = 1
    input_mask = skimage.transform.resize(canvas, output_shape=IMG_SIZE+(1,), mode='constant', preserve_range=True) 
    
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def get_data(start_index, end_index, df):
    '''
    Get data(image,mask) from datafrme and pack it into array
    '''
    x = []
    y = []
    for i in range(start_index, end_index):
        im,msk = get_image(i,df)
        x.append(im)
        y.append(msk)
    return x,y

x_clr,y_clr = get_data(0,50,empty)
x,y = get_data(0,2000,ships)
X_train = np.concatenate((np.array(x_clr),np.array(x)),axis=0)
Y_train = np.concatenate((np.array(y_clr),np.array(y)),axis=0)


X_train, Y_train = shuffle(X_train, Y_train, random_state=0)


x_test,y_test = get_data(2000,2200,ships)
X_test = np.array(x)
Y_test = np.array(y)