import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('kn1_model.pkl', 'rb'))
model = pickle.load(open('nb1_model.pkl', 'rb'))
model = pickle.load(open('dt1_model.pkl', 'rb'))
model = pickle.load(open('rf1_model.pkl', 'rb'))
model = pickle.load(open('li1_model.pkl', 'rb')) 

@app.route('/')
def home():
  
    return render_template("index2.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    _10th = float(request.args.get('_10th'))
    _12th = float(request.args.get('_12th'))
    B_tech = float(request.args.get('B_tech'))
    _7_sem = float(request.args.get('_7_sem'))
    _6_sem =float(request.args.get('_6_sem'))
    _5_sem = float(request.args.get('_5_sem'))
    final_performance = float(request.args.get('final_performance'))
    Medium = float(request.args.get('Medium'))
    
    prediction = model.predict([[_10th, _12th,B_tech,_7_sem,_6_sem,_5_sem,final_performance,Medium]])
    
        
    return render_template('index2.html', prediction_text='All Model  has predicted Placment for given Data is : {}'.format(prediction))
    if __name__=="__main__":
         app.run(debug=True)
