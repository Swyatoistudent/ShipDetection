import numpy as np
import skimage
from PIL import Image
import tensorflow as tf

IMG_SIZE = (256,256)

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


load_img = lambda dir,filename: np.array(Image.open(f"{dir}/{filename}"))


def normalize(input_image,input_mask):
    input_image = tf.cast(input_image,tf.float32) / 255.0
    input_image= tf.image.resize(input_image, IMG_SIZE,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, input_mask


def get_image(image_dir,index,df):
    '''
    Get image and mask. Resize and normalize it
    '''
    input_image = load_img(image_dir, df.loc[index, 'ImageId'])
   
    mask_pixels = rle_to_pixels(df.loc[index, 'EncodedPixels'])
    canvas = np.zeros((768,768))
    canvas[tuple(zip(*mask_pixels))] = 1
    input_mask = skimage.transform.resize(canvas, output_shape=IMG_SIZE+(1,), mode='constant', preserve_range=True) 
    
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def get_data(image_dir, start_index, end_index, df):
    '''
    Get data(image,mask) from dataframe and pack it into array
    '''
    x = []
    y = []
    for i in range(start_index, end_index):
        im,msk = get_image(image_dir,i,df)
        x.append(im)
        y.append(msk)
    return x,y