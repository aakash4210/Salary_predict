
# 98% accuracy

from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#imprt pkl file

redge_model=pickle.load(open('models/regressor_file.pkl','rb'))
Standard_scaler=pickle.load(open('models/scaler_file.pkl','rb'))


#home page
@app.route("/")
def index():
    return render_template('index.html')



#2nd page
@app.route("/saleryPredic",methods=['GET','POST'])
def predicpackage():
    if request.method == "POST":
        #taked input form web
        yearE=float(request.form.get('yearE'))
        fname=str(request.form.get('fname'))

        #transform
        new_data_scaled=Standard_scaler.transform([[yearE]])

        result=redge_model.predict(new_data_scaled)
        
        result= round(result[0], 2)
        #print("type of result :- ",type(result))
        print("trim:-",result)
        
        #return render_template('sal.html',fname=fname,result=result[0])
        return render_template('c1.html',fname=fname,result=result)
    
    else:
        return render_template('c1.html')
        


#https://gray-teacher-xgqts.pwskills.app:5000/saleryPredic

if __name__=="__main__":
    app.run(host="0.0.0.0")
