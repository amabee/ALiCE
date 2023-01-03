import keras
import nltk
import json
import pickle
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation
import random
import datetime
import webbrowser
from googlesearch import *
import requests
from ytmusicapi import YTMusic
import time
from pygame import mixer
import os
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = []
labels = []
docs_ = []
ignore = ['?' , '!' , ',' , '.']

with open("intents.json") as data_file:
    intents = json.load(data_file)

for intent in intents['intents']:
    for pattern in intent["patterns"]:
        wrd = nltk.word_tokenize(pattern)
        words.extend(wrd)
        docs_.append((wrd,intent['tag']))

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word != ignore]
words = sorted(list(set(words)))
labels = sorted(list(set(labels)))
pickle.dump(words,open('words.pkl' , 'wb'))
pickle.dump(labels,open('labels.pkl' , 'wb'))

#TRAINING CODE

training_=[]
empty_out = [0]*len(labels)

for doc in docs_:
    bag=[]
    pattern=doc[0]
    pattern=[lemmatizer.lemmatize(word.lower()) for word in pattern]

    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    
    row_output = list(empty_out)
    row_output[labels.index(doc[1])] = 1
    training_.append([bag, row_output])

random.shuffle(training_)
training_ = np.array(training_, dtype=object) #nag add kog dtype=object, if ever medyo siya buggy. try lang ug remove adto :)
training_X = list(training_[:,0])
training_Y = list(training_[:,1])

#MODEL CODE

model = Sequential()
model.add(Dense(128,activation='relu' , input_shape=(len(training_X[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_Y[0]), activation='softmax'))

adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy'])
weights = model.fit(np.array(training_X), np.array(training_Y), epochs=1000, batch_size=15, verbose=2)
model.save('model.h5',weights)

model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl' , 'rb'))
labels = pickle.load(open('labels.pkl' , 'rb'))

#Predictions Area

def cleanUp(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def createBow(sentence , words):
    sentence_words = cleanUp(sentence)
    bag=list(np.zeros(len(words)))

    for _s_ in sentence_words:
        for i,w in enumerate(words):
            if w == _s_:
                bag[i] =1
            else:
                bag[i] = 0
    return np.array(bag)

def predict_label(sentence,model):
    pre=createBow(sentence,words)
    Res_ = model.predict(np.array([pre]))[0]
    threshold = 0.8
    try:
        results = [[i,r] for i,r in enumerate(Res_) if r>threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list=[]

        for result in results:
            return_list.append({'intent':labels[result[0]], 'prob':str(result[1])})
        return return_list
    except Exception as ex:
        print(ex)

def _getResponse(return_list , intents_json):
    if len(return_list) == 0:
        tag='no response'
    else:
        tag=return_list[0]['intent']
    
    if tag== 'datetime':
        print(time.strftime("%A"))
        print(time.strftime("%d %B %Y"))
        print(time.strftime("%H:%M:%S"))

    if tag == 'google':
        search_query = input("Enter Query...")
        chrome_path = r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
        for url in search(search_query, tld="co.in", num=1, stop=1, pause=2):
            webbrowser.open("https://google.com/search?q=%s" % search_query)
        
    if tag=='weather':
        api_key='987f44e8c16780be8c85e25a409ed07b'
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        city_name = input("Enter city name: ")
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url) 
        x=response.json()
        print('Present temp.: ',round(x['main']['temp']-273,2),'celcius ')
        print('Feels Like:: ',round(x['main']['feels_like']-273,2),'celcius ')
        print(x['weather'][0]['main'])

    if tag == 'music':
        yt = YTMusic('headers_auth.json')
        playlistId = yt.create_playlist('test', 'test description')
        music_query = input("Enter Music you want to be played: ")
        search_results = yt.search(music_query)
        yt.add_playlist_items(playlistId, [search_results[0]['videoId']])
    
    if tag == 'timer':
        mixer.init()
        time_input = input("Minutes to Time...")
        time.sleep(float(time_input)*60)
        if os.path.exists('btchwtfhaha.mp3'):
            mixer.music.load('btchwtfhaha.mp3')
            mixer.music.play()
        else:
            print("The sound doesn't exist")

    intents_list = intents_json['intents']
    for i in intents_list:
        if tag == i['tag']:
            result = random.choice(i['responses'])
            return result

def response(text):
    return_list = predict_label(text,model)
    response = _getResponse(return_list, intents)
    return "ALiCE: "+ response

print("Type Something...")
while(1):
    x=input("You: ")
    print(response(x))
    if x.lower() in ["Goodbye" , "Take care" , "See you later" , "See you soon" , "I'll talk to you later" , "I'll see you tomorrow" , "I'll see you next time" , "Sayonara", "bye", 'quit']:
        break