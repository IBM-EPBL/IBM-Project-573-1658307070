from pydoc import render_doc
from distributed import connect

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import plotly
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from plotly.offline import iplot, init_notebook_mode
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import plotly
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from plotly.offline import iplot, init_notebook_mode
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2
from flask import Flask, render_template, request, url_for
from cloudant.client import Cloudant

client=Cloudant.iam('6b2497c0-c725-4938-b8a1-1c3e458f5e77-bluemix','aXe9BTFF_q8czjIKToN3ALOYUhXda38k37bW2maLgOAF', connect=True)
my_database=client.create_database('my_database')


app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/login')
def login():
   return render_template('login.html')

@app.route('/afterlogin', methods=['POST'])
def afterlogin():
   user = request.form['_id']
   passw= request.form['psw']
   print(user,passw)

   query={'_id':{'$eq': user}}

   docs= my_database.get_query_result(query)
   print(docs)
   print(len(docs.all()))

   if (len(docs.all())==0):
      return render_template('login.html')
   else:
      if((user==docs[0][0]['_id'] and passw==docs[0][0]['psw'])):
         return render_template('prediction.html')
      else:
         print('invalid user')


@app.route('/register')
def register():
   return render_template('register.html') 

@app.route('/afterreg', methods=['POST'])
def afterreg():
   username=request.form['_id']
   password=request.form['psw']
   data={
   '_id':username,
   # 'name':x[0],
   'psw':password  
   }
   print(data)

   query={'_id': {'$eq':username}}

   docs=my_database.get_query_result(query)
   print(docs)
   print(len(docs.all()))

   if(len(docs.all())==0):
      url=my_database.create_document(data)
      return render_template('prediction.html')
   else:
      return render_template('register.html')

   

@app.route('/prediction')
def prediction():
   return render_template('prediction.html')   

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(f.filename)
      file_name=f.filename
      
      model=load_model("retina_weights.hdf5")
      prediction = []
      
      image = []
     
      labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}
      
      img= PIL.Image.open(file_name)
 
      img = img.resize((256,256))
  
      image.append(img)
 
      img = np.asarray(img, dtype= np.float32)
 
      img = img / 255
  
      img = img.reshape(-1,256,256,3)
  
      predict = model.predict(img)
  
      predict = np.argmax(predict)
 
      prediction.append(labels[predict])

      print(prediction)
      
      return render_template("output.html", prediction=prediction[0])
		
if __name__ == '__main__':
   app.run(debug = True)
   