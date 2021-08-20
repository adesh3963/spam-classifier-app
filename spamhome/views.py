from django.shortcuts import render,HttpResponse
from spamhome.models import  Messeges

### importing data for my spam classifier
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle

### text preprocessing
lemmatizer=WordNetLemmatizer()
def tprep(text):
    remove=re.sub('[^a-zA-Z]',' ',text)
    remove=remove.lower()
    words=remove.split()
    lemma_words=[lemmatizer.lemmatize(wrd) for wrd in words if wrd not in set(stopwords.words("english"))]
    text_prep= [' '.join(lemma_words)]
    return text_prep

### word vectorization


### prediction


# Create your views here.
def clf(request):
    
    if request.method=="POST":
        print("data comming")
        text=request.POST["message"]
        inst=Messeges(text=text)
        inst.save()
        print("load successfull")


        length_of_text= np.array(len(text))

        msg=tprep(text)

        loaded_vertorizer = pickle.load(open("data_and_pkl/vetorizer.pkl", 'rb'))

        text_transform=loaded_vertorizer.transform(msg).toarray()

        
        text_transform=np.append(text_transform,length_of_text) 

        text_transform= [text_transform]

        loaded_model = pickle.load(open("data_and_pkl/spam_clf_model.pkl", 'rb'))
        result = loaded_model.predict(text_transform)

        print(result)
        context2={"prediction":"Spam"}
        context={"prediction":"Not spam"}
        if result==0:
            return render(request,"result.html",context)
        else:
            return render(request,"result.html",context2)
    else:
        return render(request,"base.html")

def about(request):
    return render(request,"about.html")  


    
