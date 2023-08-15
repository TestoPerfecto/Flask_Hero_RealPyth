#import Flask 
from flask import Flask, render_template, request
import numpy as np
import joblib
import sklearn
#create an instance of Flask
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        #get form data
        tv = request.form.get('tv')
        radio = request.form.get('radio')
        newspaper = request.form.get('newspaper')
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(tv, radio, newspaper)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
        except ValueError:
            return "Please Enter valid values"
        pass        
    pass
def preprocessDataAndPredict(tv, radio, newspaper):
    #put all inputs in array
    test_data = [tv, radio, newspaper]
    print(test_data)
    #convert value data into numpy array and type float
    test_data = np.array(test_data).astype(np.float64) 
    #reshape array
    test_data = test_data.reshape(1,-1)
    print(test_data)
    #open file
    file = open("deploy_flask_gcp_model.pkl","rb")
    #load trained model
    trained_model = joblib.load(file)
    #predict
    prediction = trained_model.predict(test_data)
    return prediction
    pass
if __name__ == '__main__':
    app.run(debug=True)