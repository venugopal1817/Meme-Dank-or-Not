#!/usr/bin/env python3
import numpy as np
import cv2 
from keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

model = VGG16

class image_features:
    def __init__(self, image,model):
        self.VGG_img_feature = model #VGG16()
        self.image = image

    def objects(self):
        #extracting objects from image using VGG16
        
        pix = np.asarray(self.image)
        pix = pix.astype('float32')
        pix.resize(224,224,3)
        pix = np.expand_dims(pix, axis=0)
        pix = preprocess_input(pix)
        prediction_img = self.VGG_img_feature.predict(pix)
        final_label = decode_predictions(prediction_img, top=3)
        return final_label
