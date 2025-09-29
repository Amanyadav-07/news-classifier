from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load pipeline dictionary (make sure your model is saved as a dict)
pipeline = joblib.load("news_classifier_Logistic Regression.pkl")

model = pipeline["model"]
vectorizer = pipeline["vectorizer"]
label_encoder = pipeline["label_encoder"]

# Download resources once
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["news_text"]

        if not text.strip():
            return render_template("index.html", prediction="⚠️ Please enter some text", input_text=text)

        clean = clean_text(text)
        vectorized = vectorizer.transform([clean])
        pred = model.predict(vectorized)
        category = label_encoder.inverse_transform(pred)[0]
        return render_template("index.html", prediction=category, input_text=text)

if __name__ == "__main__":
    app.run(debug=True)
