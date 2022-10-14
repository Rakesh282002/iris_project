from flask import Flask, render_template,url_for,request,redirect
import pickle
import numpy as np
model=pickle.load(open("iris.pkl",'rb'))
app = Flask(__name__)
@app.route('/',methods=["GET","POST"])
def home():
    return render_template("iris_web.html")
@app.route('/home1',methods=['POST'])
def home1():
    data1=request.form['sl']
    data2=request.form['sw']
    data3=request.form['pl']
    data4=request.form['pw']
    data1 = np.asarray(data1, dtype='float64')
    data2 = np.asarray(data2, dtype='float64')
    data3 = np.asarray(data3, dtype='float64')
    data4 = np.asarray(data4, dtype='float64')
    arr=np.array([[data1,data2,data3,data4]])
    pred=model.predict(arr)
    return render_template("after_iris_web.html",data=pred)
if __name__=="__main__":
    app.run(debug=True)