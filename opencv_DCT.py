import threading
import io
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import IPython
import numpy as np
import cv2


def myfunc(i):
    pass # do nothing



img = np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image') # create win with win name


cv2.createTrackbar('frequency','image',0,640,myfunc)


"""
cv2.createTrackbar('value', # name of value
                   'title', # win name
                   0, # min
                   640, # max
                   myfunc) # callback func
"""


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  32)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 32)

def get_image(frame):
    
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = np.array(img_grey, dtype=np.float)
    return img
 
def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = np.array(img, dtype='uint8')
    return img


while(True):

    ret, frame = cap.read()
    if not ret: continue
    frame=get_image(frame)
    pixels = frame
    dct_size = pixels.shape[0]
    dct = get_2D_dct(pixels)
    reconstructed_images = []
    for ii in range(dct_size):
        dct_copy = dct.copy()
        dct_copy[ii:,:] = 0
        dct_copy[:,ii:] = 0
        # Reconstructed image
        r_img = get_2d_idct(dct_copy);
        reconstructed_image = get_reconstructed_image(r_img);
        # Create a list of images
        reconstructed_images.append(reconstructed_image);
        v = cv2.getTrackbarPos('frequency',  # get the value
                       'image')  # of the win

    ## do something by using v
   
    cv2.imshow('image', reconstructed_images[int(v/10)])  # show in the win
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        cap.release()
        cv2.destroyAllWindows()




    