import os
from tokenize import tokenize
import pandas as pd 
import numpy as np 
import flask
import pickle
from flask import Flask, render_template, request
import tensorflow as tf 
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import re
 

app=Flask(__name__)

def normalize(data):
    normalized = []
    for i in data:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

@app.route('/')
def index():
 return flask.render_template('index.html')
def ValuePredictor(to_predict_text):
 result=""
#  loaded_model = pickle.load(open("model.pkl","rb"))
 tokenizer = pickle.load(open("tokenizer_pkl","rb"))
 loaded_model=keras.models.load_model("news_prediction.h5")
#  max_vocab = 10000
#  tokenizer = Tokenizer(num_words=max_vocab)


 new_t= tokenizer.texts_to_sequences(to_predict_text)
 new_p=tf.keras.preprocessing.sequence.pad_sequences(new_t, padding='post', maxlen=256)
#  result = loaded_model.predict(to_predict_text)
 print(new_p)
 if (loaded_model.predict(new_p)>=0.5).astype(int)==0:
     result="the input is fake news"
 else:
     result="the input is real news"
 return result

@app.route('/predict',methods = ['POST'])

def result():
 if request.method == 'POST':
    to_predict_text = request.form.get("name")
    r=""
    r=to_predict_text
    b=[r]
    n=normalize(b)
    print(n)
    result = ValuePredictor(n)
    prediction = result
 return render_template('prediction.html',prediction=prediction)

 
if __name__ == "__main__":
 app.run(debug=True)