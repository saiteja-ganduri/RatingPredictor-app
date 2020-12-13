from flask import Flask ,render_template ,request
import numpy as np
import joblib
# from sklearn.externals import joblib
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predictcall",methods=['POST'])
def predictcall():
    if request.method == 'POST':
        x=request.form['message']
        sam=[x]
        vector = joblib.load('tfidf_vectorizer.sav')
        ridge = joblib.load('Ridge_model.sav')
        text = vector.transform(sam)
        result = np.round(ridge.predict(text))
        result=str(result).strip('[.]')
        return render_template('index.html',result=result)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)   
