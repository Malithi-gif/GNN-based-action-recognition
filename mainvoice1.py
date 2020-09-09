import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from keras.models import load_model
import array
import numpy as np
from stt import SpeechToCnv as speechToText
import os    
import dialogflow
import requests
import json
import pusher
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import time
import threading
import time
import tensorflow as tf
from ann_visualizer.visualize import ann_viz


cred = credentials.Certificate("./mithsara-uwjvun-firebase-adminsdk-y0q2j-5b631dce70.json")
 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'mithsara-uwjvun-f9da593d7361.json'
db = firestore.client()
doc_ref = db.collection(u'data').document(u'one')
doc_ref.set({
    u'value': u'2',
})
    

dataStore=[]


with open("intents1.json") as file:
    data = json.load(file)

# print(data)

# with open("store.pickle", "rb") as f:
#     words, labels, training, output = pickle.load(f)


words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output) 

with open("store.pickle", "wb") as f:
     pickle.dump((words, labels, training, output), f)


tensorflow.reset_default_graph()
graph=tf.Graph()
with tf.Graph().as_default():

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


# # ann_viz(model, title="Artificial neural network")

# sess=tf.Session(graph=graph)
# writer=tf.train.summary.FileWriter('tmp/tensorboard_log',sess.graph)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

st = speechToText()
def chat():
    
    print("Start talking with the bot (type quit to stop)!")
    for x in range(0,12):
        
        # inp = input("You: ")
        # if inp.lower() == "quit":
        #     break
        time.sleep(14)
        print("talk to robot")
        inp = st.convert()
        print("converting speech")
        print(inp) 
   



        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

        if inp.find("special") > -1:
                dataStore.append(0)
                dataStore.append(1)
                dataStore.append(0)
                print(dataStore[0])
        elif inp.find("general degree") > -1:
                dataStore.append(1)
                dataStore.append(0)
                dataStore.append(0)
                print(dataStore[1])
        elif inp.find("extended") > -1:
                dataStore.append(0)
                dataStore.append(0)
                dataStore.append(1)
                print(dataStore[2])
        elif inp.find("first class") > -1:
                dataStore.append(4)
                print(dataStore[3])
        elif inp.find("second upper") > -1:
                dataStore.append(3) 
                print(dataStore[3])
        elif inp.find("second up") > -1:
                dataStore.append(3)
                print(dataStore[3])
        elif inp.find("second Appa") > -1:
                dataStore.append(3)
                print(dataStore[3])
        elif inp.find("second lower") > -1:
                dataStore.append(2)
                print(dataStore[3])
        elif inp.find("general") > -1:
                dataStore.append(1)
                print(dataStore[3])
        elif inp.find("software engineering") > -1:
                dataStore.append(1)
                dataStore.append(0)
                dataStore.append(0)
                print(dataStore[4])
                print(dataStore[5])
                print(dataStore[6])
        elif inp.find("management") > -1:
                dataStore.append(0)
                dataStore.append(1)
                dataStore.append(0)
                print(dataStore[4])
                print(dataStore[5])
                print(dataStore[6])
        elif inp.find("none") > -1:
                print("kkkk")
                dataStore.append(0)
                dataStore.append(0)
                dataStore.append(1)
                print(dataStore[4])
                print(dataStore[5])
                print(dataStore[6])
        elif inp.find("Nan") > -1:
                print("kkkk")
                dataStore.append(0)
                dataStore.append(0)
                dataStore.append(1)
                print(dataStore[4])
                print(dataStore[5])
                print(dataStore[6])
        elif inp.find("not follow") > -1:
                print("kkkk")
                dataStore.append(0)
                dataStore.append(0)
                dataStore.append(1)
                print(dataStore[4])
                print(dataStore[5])
                print(dataStore[6])
        elif inp.find("extremely well") > -1:
                dataStore.append(1)
                print(dataStore[7])
        elif inp.find("very well") > -1:
                dataStore.append(1)
                print(dataStore[7])
        elif inp.find("little bit") > -1:
                dataStore.append(0)
                print(dataStore[7])
        elif inp.find("I can") > -1:
                dataStore.append(1)
                print(dataStore[8])
        elif inp.find("I can't") > -1:
                dataStore.append(0)
                print(dataStore[8])
        elif inp.find("a plus") > -1:
                dataStore.append(4)
                print(dataStore[9])
        elif inp.find("a") > -1:
                dataStore.append(4)
                print(dataStore[9])
        elif inp.find("a mine") > -1:
                dataStore.append(4)
                print(dataStore[9])
                print(dataStore[9])
        elif inp.find("b plus") > -1:
                dataStore.append(3)
        elif inp.find("b") > -1:
                dataStore.append(3)
                print(dataStore[9])
        elif inp.find("b mine") > -1:
                dataStore.append(2)
                print(dataStore[9])
        elif inp.find("c plus") > -1:
                dataStore.append(2)
                print(dataStore[9])
        elif inp.find("c") > -1:
                dataStore.append(1)
                print(dataStore[9])
        elif inp.find("c mine") > -1:
                dataStore.append(1)
                print(dataStore[9])
                print(dataStore[9])
        elif inp.find("yes") > -1:
                dataStore.append(1)
                print(dataStore[10])
        elif inp.find("no") > -1:
                dataStore.append(0)
                print(dataStore[10])
        elif inp.find("I have") > -1:
                dataStore.append(1)
                print(dataStore[11])
        elif inp.find("I haven't") > -1:
                dataStore.append(0)
                print(dataStore[11])
        elif inp.find("high") > -1:
                dataStore.append(3)
                print(dataStore[12])
        elif inp.find("moderate") > -1:
                dataStore.append(2)
                print(dataStore[12])
        elif inp.find("low") > -1:
                dataStore.append(1)
                print(dataStore[12])
        elif inp.find("I did") > -1:
                dataStore.append(1)
                print(dataStore[13])
        elif inp.find("ID") > -1:
                dataStore.append(1)
                print(dataStore[13])
        elif inp.find("I did not") > -1:
                dataStore.append(0)
                print(dataStore[13])
       
        elif inp.find("more than one") > -1:
                dataStore.append(2)
                print(dataStore[14])
        elif inp.find("one publication") > -1:
                dataStore.append(1)
                print(dataStore[14])
        elif inp.find("no publications") > -1:
                dataStore.append(0)
                print(dataStore[14])
        elif inp.find("haven't publications") > -1:
                dataStore.append(0)
                print(dataStore[14])


        print(dataStore)


        try:
            st.speak(random.choice(responses))
            db = firestore.client()
            doc_ref = db.collection(u'data').document(u'one')
            doc_ref.set({
                u'value': u'1',
            })
        except AssertionError as error:
            st.speak("I did not here you")
            print(error)
        # return np.array(dataStore)
chat()


print(dataStore)
model1 = load_model('new_model2.h5')
                    # loaded_model.layers[0].input_shape #(None, 160, 160, 3)
                    # result = loaded_model.predict([[0, 1, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 2, 4, 3, 4, 1, 0, 0, 0, 0]])
arrTest1 = np.array([dataStore]);
scores = model1.predict(arrTest1)
scores1 = scores.argmax(axis=-1)
print(scores1)

if scores1==0:
    print("you can become as a software engineer")
    st.speak("I think software engineer is most convenient to you")
    db = firestore.client()
    doc_ref = db.collection(u'data').document(u'one')
    doc_ref.set({
        u'value': u'1',
    })     

if scores1==1:
    print("you can become as Businees analysist")
    st.speak("I think Businees analysist is most convenient to you")
    db = firestore.client()
    doc_ref = db.collection(u'data').document(u'one')
    doc_ref.set({
        u'value': u'1',
    })     

if scores1==2:
    print("you can become as Quality assurer")
    st.speak("I think Quality assurance engineer is most convenient to you")
    db = firestore.client()
    doc_ref = db.collection(u'data').document(u'one')
    doc_ref.set({
        u'value': u'1',
    })  

if scores1==3:
    print("I think you can best in Academic field")

    st.speak("I think Academic field is most convenient to you")
    db = firestore.client()
    doc_ref = db.collection(u'data').document(u'one')
    doc_ref.set({
        u'value': u'1',
    }) 
      


# def predict():



        

# #           
#         model1 = load_model('new_model.h5')
#                     # loaded_model.layers[0].input_shape #(None, 160, 160, 3)
#                     # result = loaded_model.predict([[0, 1, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 2, 4, 3, 4, 1, 0, 0, 0, 0]])
#         arrTest1 = np.array([[0, 1, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 2]]);
#         scores = model1.predict(arrTest1)
#         scores1 = scores.argmax(axis=-1)
#         print(scores1)

# predict()  
n