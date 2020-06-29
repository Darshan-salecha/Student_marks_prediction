# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
from flask import Flask,request,render_template

app=Flask(__name__)

model=joblib.load("Student_marks_predtictor_model.pkl")
df=pd.DataFrame()


@app.route('/')

def  home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    global df
    input_features=[int(x) for x in request.form.values()]
    feature_value=np.array(input_features)
    
    if input_features[0]<0 or input_features[0]>24:
        return render_template('index.html',prediction_text='Please write proper Hours')
    
    output=model.predict([feature_value])[0][0].round(2)
    if output>100:
        output=100 #marks cannot be more than 100
    df=pd.concat([df,pd.DataFrame({'input_hour':input_features,'Predicted_output':[output]})],ignore_index=True)
    print(df)
    
    df.to_csv('data_from_app.csv')
    return render_template('index.html',prediction_text='you will get [{}%] marks, when you do study [{}] hours per day'.format(output,int(feature_value[0])))

if __name__=='__main__':
    app.run()

                           