#UPDATE
from fastapi import FastAPI,Request,Form,File,UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import base64
from fastapi.staticfiles import StaticFiles
#import shutil

import numpy as np
import os
import pickle
import string

#UPDATE   -- required To resolve -- AttributeError: module 'collections' has no attribute 'Callable'
import collections
collections.Callable = collections.abc.Callable

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model


import logging

#level = logging.DEBUG
level = logging.INFO
#level = logging.WARNING
logger = logging.getLogger()
logger.setLevel(level)

#logging.debug('Debug message')
#logging.info('Info message')
#logging.warning('Warning message')
#logging.error('Error message')
#logging.critical('Critical message')

logging.basicConfig(filename='ImageCaption.log', filemode='w', format='%(asctime)s :: %(name)s - %(levelname)s - %(message)s')
logging.warning('This message will get logged on to a file')

#UPDATE
#pickle_features = pickle.load(open('features.pkl', 'rb'))
#FEATURES_PATH = r'C:\B20\Dataset\Models\VGG16\VGG16_features.pkl'
FEATURES_PATH = r'.\Models\VGG16\VGG16_features.pkl'
pickle_features_1 = pickle.load(open(FEATURES_PATH, 'rb'))

#UPDATE
#model_path = r'C:\B20\Dataset\Models\VGG16\VGG16_LSTM_ImageCaptionModel.h5'
model_path = r'.\Models\VGG16\VGG16_LSTM_ImageCaptionModel.h5'
model_1 = tf.keras.models.load_model(model_path)

# UPDATE
#FEATURES_PATH = r'C:\B20\Dataset\Models\Inception_v3\InceptionV3features.pkl'
FEATURES_PATH = r'.\Models\Inception_v3\InceptionV3features.pkl'
pickle_features_2 = pickle.load(open(FEATURES_PATH, 'rb'))

#UPDATE
#model_path = r'C:\B20\Dataset\Models\Inception_v3\InceptionV3ImageCaption.h5'
model_path = r'.\Models\Inception_v3\InceptionV3ImageCaption.h5'
model_2 = tf.keras.models.load_model(model_path)  

#UPDATE
#CAPTIONS_DIR = r'C:\B20\Dataset\Flicker8k_Text\Flickr8k.token.txt'
CAPTIONS_DIR = r'.\Flicker8k_Text\Flickr8k.token.txt'

with open(os.path.join(CAPTIONS_DIR), 'r') as f:
    captions_doc = f.read()

mapping = {}
for line in (captions_doc.split('\n')):
    tokens = line.split('\t')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

def clean(mapping):
    logging.info('clean() - Entry')
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            caption = caption.translate(str.maketrans('','',string.punctuation))
            # delete digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
    logging.info('clean() - Exit')

clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

max_length = max(len(caption.split()) for caption in all_captions)

def idx_to_word(integer, tokenizer):
    logging.debug('idx_to_word() - Entry')
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    logging.debug('idx_to_word() - Exit')
    return None

def predict_caption(model, image, tokenizer, max_length):
    logging.info('predict_caption() - Entry')
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        logging.debug('predict_caption() - for loop# %i', i)
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    logging.info('predict_caption() - Exit')
    return in_text

def generate_caption(image_name, model, pickle_features):
    logging.info('generate_caption() for %s - Entry', image_name)
    #logging.info('generate_caption() - Entry', image_name)
    image_id = image_name.split('.')[0]
    logging.info('Image ID is =', image_id)
    # predict the caption
    y_pred = predict_caption(model, pickle_features[image_id], tokenizer, max_length)
    logging.info('generate_caption() - Exit')
    return y_pred

app = FastAPI()
templates = Jinja2Templates(directory = 'templates/')

@app.get("/predict/", response_class=HTMLResponse)
async def upload(request: Request):
   return templates.TemplateResponse("test.html", {"request": request})
   
@app.post("/predicted/")
async def create_upload_file(request:Request, file: UploadFile = File(...)):
        
        logging.info('create_upload_file() - Entry')
        logging.info('File Name is %s', file.filename)
        ImageID = file.filename
                    
        logging.info('Start Caption Generation using "VGG16 + CNN + LSTM"')
        result1 = generate_caption(ImageID, model_1, pickle_features_1) 
        logging.info('End Caption Generation using "VGG16 + CNN + LSTM"')
                                          
        logging.info('Start Caption Generation using "Inception_v3 + CNN + LSTM"')
        result2 = generate_caption(ImageID, model_2, pickle_features_2)
        logging.info('End Caption Generation using "Inception_v3 + CNN + LSTM"')
        
        data = file.file.read()
        file.file.close()
        # encoding the image
        encoded_image = base64.b64encode(data).decode("utf-8")
        
        logging.info('create_upload_file() - Exit')
        
        return templates.TemplateResponse('test.html',context = {'request':request,'result1':result1, 'result2':result2,'pic':encoded_image})
        
        
#https://geekpython.in/displaying-images-on-the-frontend-using-fastapi
#---------------Following Code to display any static image from local PC folders
#---------------"/local_image_folder" - is a sub-path on which the sub-application will be mounted.
#--------------"Images" is sub-folder (with images) located in root folder where main.py is present
#app.mount("/local_image_folder", StaticFiles(directory="Images"))

#--------------"C:\B20\Dataset" is folder, where images are located
app.mount("/local_image_folder", StaticFiles(directory="C:\B20\Dataset"))

@app.get("/staticimage", response_class=HTMLResponse)
#@app.get("/", response_class=HTMLResponse)

def serve_2():
    return """
    <html>
        <head>
            <title></title>
        </head>
        <h1>Static Image Local PC Folder {{directory}}</h1>
        <body>
        <!--<img src="local_image_folder/10815824_2997e03d76.jpg">-->
        <img src="local_image_folder/33108590_d685bfe51c.jpg">
        </body>
    </html>
    """
