from flask import Flask,render_template,request,Response
import pickle
from werkzeug.utils import redirect
import os
import sklearn

app=Flask(__name__)
@app.route('/')
def shashi():
    return render_template('index.html')
@app.route('/submit',methods=['POST','GET'])
def getdetails():
    if request.method=='GET':
        p=float(request.form['Pregnancies'])
        g=float(request.form['Glu'])
        b=float(request.form['Bp'])
        skinthickness=float(request.form['st'])
        insulin=float(request.form['Insulin'])
        bmi=float(request.form['BMI'])
        dpf=float(request.form['dpf'])
        test=[p,g,b,skinthickness,insulin,bmi,dpf]
        filename = 'diabatesfinalized_model'
        loaded_model = pickle.load(open(filename, 'rb'))
        result=loaded_model.predict([test])
        if result==0:
            p=0
            return render_template('predict.html',r=p)  
        else:
            p=1
            return render_template('predict.html',r=p)

if __name__=='__main__':
    app.run(debug=True)
