import numpy as np
import flask
import keras
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)


app = Flask(__name__)
model = load_model('D:/ML/ann_nepali.h5')

##CNN part
##model = load_model('D:/ML/nepali_cnn_model/cnn.h5')

graph = tf.get_default_graph()


@app.route("/")
def index():
    return flask.render_template('index.html')
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        datas = []
        my_data = pd.read_csv("D:\ML\labels.csv",delimiter=",")

       #ANN Paer
        file = request.files['image']
        test_image = misc.imread(file, flatten=True)
        datas.append(test_image.flatten())
        x = np.array(datas)


        ##CNN part
        #test_image = misc.imread(file)
        #datas.append(test_image)
        #x = np.array(datas)
        #X=np.array(x).reshape(-1,36,36,1)


        with graph.as_default():
            prediction = model.predict_classes(x)
            prob = model.predict_proba(x)
        #
        # # squeeze value from 1D array and convert to string for clean return
        #int coverts that string to integer
        label = int(np.squeeze(prediction))
        accuracy=(np.amax(prob))*100
        answer = str(round(accuracy, 2))

        # # switch for case where label=10 and number=0
        #my_data.iloc[label,:].values[1]
        return render_template('index.html',label = my_data.iloc[label,:].values,prob=answer)

if __name__ == '__main__':
    app.run(debug=True)
