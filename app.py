from flask import Flask, render_template, request
import os
import pickle
from tqdm import tqdm
import tensorflow.compat.v1 as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

with open('captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()

mapping = {}

for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    
    image_id = image_id.split('.')[0]
    
    caption = " ".join(caption)
    
    if image_id not in mapping:
        mapping[image_id] = []
    
    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)

model = tf.keras.models.load_model('best_model.h5')

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        
        if word is None:
            break

        in_text += " " + word
        
        if word == 'endseq':
            break

    return in_text

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', caption="No image uploaded!")

        image_file = request.files['image']
        if image_file.filename == '':
            return render_template('index.html', caption="No image selected!")

        # Save the image to a temporary location
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        image = load_img(image_path, target_size=(224, 224))

        image = img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        image = preprocess_input(image)

        feature = vgg_model.predict(image, verbose=0)

        # Call the caption generation function
        caption = predict_caption(model, feature, tokenizer, max_length)

        caption = caption.replace('startseq', '').replace('endseq', '').strip()

        caption = caption[0].capitalize() + caption[1:]

        # Remove the temporary image file
        os.remove(image_path)

    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)

    
