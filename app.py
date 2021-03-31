from flask import Flask, render_template, redirect, request, url_for
from ml_utils import Predicter
app = Flask("Spam Classification App")

@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template("home.html", validation_mssg="")

@app.route("/predict", methods=['POST'])
def get_prediction():
    #print(list(request.form.keys()))
    #print(request.form["text"])
    text_val = request.form["text"]
    prediction= Predicter.run(text_val)
    return render_template("predict.html", validation_mssg="", classifier_result=prediction, text_value=text_val)





if __name__=="__main__":
    app.run(debug=True)