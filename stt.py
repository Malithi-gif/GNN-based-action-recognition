import speech_recognition as sr
from gtts import gTTS
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


class SpeechToCnv:
    cred = credentials.Certificate("./mithsara-uwjvun-firebase-adminsdk-y0q2j-5b631dce70.json")

    default_app = firebase_admin.initialize_app(cred)

    db = firestore.client()
    print("stt")
    def __init__(self):
        self.r = sr.Recognizer()

    def listen(self):
        db = firestore.client()
        doc_ref = db.collection(u'data').document(u'one')

        doc_ref.set({
          u'value': u'0',
        })
        
        print("start talking")
        with sr.Microphone() as source:
            self.audio = self.r.listen(source,phrase_time_limit = 5)


    def convert(self):

        self.listen()
        # with open("voice.wav","wb") as f:
        #     f.write(self.audio.get_wav_data())

        retValue = ""
        try:
            retValue =  self.r.recognize_google(self.audio)
        except:
            retValue = ""
        
        return retValue

    def speak(self, txt):
        db = firestore.client()
        doc_ref = db.collection(u'data').document(u'one')

        doc_ref.set({
            u'value': u'1',
        })

        
        mytext = txt
        language = 'en'
        myobj = gTTS(text=mytext, lang=language, slow=False) 
        myobj.save("welcome.mp3") 
        os.system("welcome.mp3") 