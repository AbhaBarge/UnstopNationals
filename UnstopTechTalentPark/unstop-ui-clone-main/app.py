import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import tensorflow as tf
from keras.models import load_model
from flask_cors import CORS
import json


app = Flask(__name__)



global model, graph 
graph = tf.compat.v1.get_default_graph()
model = load_model('score_predictor.h5')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response


@app.route("/")
def jobportal():
    return render_template("job-portal.html")

@app.route('/predict', methods=['OPTIONS'])
def options():
    return '', 200

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        user = data.get('username')

    if user is None:
        app.logger.warning(f"Username {user} not found in the dataset")
        return "User doesn't exist"
        
    else :
        if user.strip() == "":
            app.logger.warning("No username provided in the form")
            return "Nothing entered in the input"
    
        df = pd.read_csv("UnstopData.csv")
        usn = df.index[df.username == user]
        
        if len(usn) > 0:
            row = usn[0]
            df2 = df[['easySolved', 'medSolved', 'diffSolved', 'enagementHrs', 'enagementMins']].iloc[row]
            newdata = np.array(df2).reshape((1,5))
            weight = np.array([0.7, 0.85, 0.9, 0.4, 0.3]).reshape((1,5))
        
            predicted_score = model.predict(newdata * weight)
            predicted_score = predicted_score + 22
            
            while predicted_score > 99:
                predicted_score -= np.random.randint(2, 5)
            
            
            print(f"Predicted score for {user}: {predicted_score[0][0]}")
            newsc = predicted_score[0][0]
            print(f"Predicted score for {user}: {newsc}")
            return str(newsc)
        
        else:
            print(f"Username {user} not found in the dataset")
            return "User doesn't exist"
            



'''  
def predict():
    user = request.form.get('usernameInput')
    df = pd.read_csv("UnstopData.csv")
    usn = df.index[df.username==user]
    if(len(usn)>0):
        row = usn[0]
        df2 =  df[['easySolved','medSolved','diffSolved','enagementHrs','enagementMins']].iloc[row]
        newdata = np.array(df2)
        weight = np.array([0.7, 0.85, 0.9, 0.4, 0.3])
        with graph.as_default():
            predicted_score = model.predict(newdata * weight)
            predicted_score = predicted_score+22
            while predicted_score>99:
                predicted_score -= np.random.randint(2,5)
            print(predicted_score)
            return render_template("result.html", prediction = predicted_score)
        
    else:
        print(-1)
        return render_template("result.html", prediction = -1)
'''
    


if __name__ =='__main__':
    app.run(host='0.0.0.0')







