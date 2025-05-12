from flask import Flask,render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html',prediction_text='')

@app.route('/prediction',methods=['POST'])
def predict():
    sl=float(request.form["sepal_length"])
    sw=float(request.form["sepal_width"])
    pl=float(request.form["petal_length"])
    pw=float(request.form["petal_width"])
    input_data=np.array([[sl,sw,pl,pw]])
    model=pickle.load(open('iris_model.pkl','rb'))
    result=model.predict(input_data)
    

    return render_template('index.html',prediction_text='The predicted class is: {}'.format(result[0]))
if __name__ == '__main__':
    app.run(debug=True)        
