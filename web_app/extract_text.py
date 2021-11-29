#!/usr/bin/env python3
#function to extract and clean text
#installing pytesseract

import cv2 
import re
import nltk
from nltk.corpus import stopwords
import tensorflow_text as text
import pytesseract

nltk.download('punkt')
nltk.download('stopwords')

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def txt_extraction_clean(urls):
    image = io.imread(urls)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) 
    alpha_To_extract = r"--oem 3 --psm 11 -c tessedit_char_whitelist= 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '"
    txt_data = pytesseract.image_to_string(image, lang='eng', config=alpha_To_extract)
   
    #cleaning text
    
    txt_data = re.sub('[^A-Za-z]',' ',urls).lower()    #Removing all characters from text except alphabets
    words = word_tokenize(txt_data)
    stopWords = set(stopwords.words('english'))
    words = [w for w in words if w not in stopWords and len(w)>3]
    txt_data = ' '.join(words)
    return text