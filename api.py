from flask import Flask,request
from flask_cors import CORS
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_curve, classification_report
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("FinalData.csv")
df.head()
newdf=df.drop(['Stages','History','Patient','ControlledDiet','TakeMedication'], axis = 1)
le = preprocessing.LabelEncoder()
df2=(newdf.apply(le.fit_transform))
result = pd.concat([df, df2], axis=1, join="inner")
N=13
finalresult = result.iloc[: , N:]
target = finalresult.Stages
inputs = finalresult.drop('Stages',axis='columns')
x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def myconverter(o):
    if isinstance(o, np.float32):
        return float(o)

@app.route("/rf",methods=["GET","POST"])
def rf():
    
    gender=request.json["gender"]
    age=request.json["age"]
    headache=request.json["headache"]
    breadth=request.json["breadth"]
    visual=request.json["visual"]
    nose=request.json["nose"]
    blood=request.json["blood"]
    ra_nge=request.json["range"]
    dRange=request.json["dRange"]

    model = RandomForestClassifier(n_estimators=50)
    model.fit(x_train,y_train)
    model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    accuracy_score(y_test, y_pred)*100
    pred=model.predict([[gender,age,headache,breadth,visual,nose,blood,ra_nge,dRange]])
    print(pred)
   
    
    return {"Success":200,"prediction":pred[0]}

@app.route("/boost",methods=["GET","POST"])
def boost():
    gender=request.json["gender"]
    age=request.json["age"]
    headache=request.json["headache"]
    breadth=request.json["breadth"]
    visual=request.json["visual"]
    nose=request.json["nose"]
    blood=request.json["blood"]
    ra_nge=request.json["range"]
    dRange=request.json["dRange"]

    AdaModel = AdaBoostClassifier(n_estimators=100, learning_rate=1)
    model = AdaModel.fit(x_train,y_train)
    pred=model.predict([[gender,age,headache,breadth,visual,nose,blood,ra_nge,dRange]])
    print(pred)

    return {"Success":200,"prediction":pred[0]}



@app.route("/naive",methods=["GET","POST"])
def naive():
    gender=request.json["gender"]
    age=request.json["age"]
    headache=request.json["headache"]
    breadth=request.json["breadth"]
    visual=request.json["visual"]
    nose=request.json["nose"]
    blood=request.json["blood"]
    ra_nge=request.json["range"]
    dRange=request.json["dRange"]

    naive = MultinomialNB()
    model = naive.fit(x_train,y_train)
    pred=model.predict([[gender,age,headache,breadth,visual,nose,blood,ra_nge,dRange]])
    print(pred)

    return {"Success":200,"prediction":pred[0]}

@app.route("/svm",methods=["GET","POST"])
def svm():
    gender=request.json["gender"]
    age=request.json["age"]
    headache=request.json["headache"]
    breadth=request.json["breadth"]
    visual=request.json["visual"]
    nose=request.json["nose"]
    blood=request.json["blood"]
    ra_nge=request.json["range"]
    dRange=request.json["dRange"]

    svm = SVC(probability=True)
    model = svm.fit(x_train,y_train)
    pred=model.predict([[gender,age,headache,breadth,visual,nose,blood,ra_nge,dRange]])
    print(pred)

    return {"Success":200,"prediction":pred[0]}

@app.route("/dt",methods=["GET","POST"])
def dt():
    gender=request.json["gender"]
    age=request.json["age"]
    headache=request.json["headache"]
    breadth=request.json["breadth"]
    visual=request.json["visual"]
    nose=request.json["nose"]
    blood=request.json["blood"]
    ra_nge=request.json["range"]
    dRange=request.json["dRange"]

    dt = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
    model = dt.fit(x_train,y_train)
    pred=model.predict([[gender,age,headache,breadth,visual,nose,blood,ra_nge,dRange]])
    print(pred)

    return {"Success":200,"prediction":pred[0]}


@app.route("/lr",methods=["GET","POST"])
def lr():
    gender=request.json["gender"]
    age=request.json["age"]
    headache=request.json["headache"]
    breadth=request.json["breadth"]
    visual=request.json["visual"]
    nose=request.json["nose"]
    blood=request.json["blood"]
    ra_nge=request.json["range"]
    dRange=request.json["dRange"]

    lr =  LogisticRegression()
    model = lr.fit(x_train,y_train)
    pred=model.predict([[gender,age,headache,breadth,visual,nose,blood,ra_nge,dRange]])
    print(pred)

    return {"Success":200,"prediction":pred[0]}

if __name__=="__main__":
    app.run(host='0.0.0.0', port=port, debug=True)
