import math
import os

from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import re
from Stemmer import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from joblib import dump, load

app = Flask(__name__)
app.current_model = None


def text_cleaner(text):
    text = text.lower()
    stemmer = Stemmer('russian')
    text = ' '.join(stemmer.stemWords(text.split()))
    text = re.sub(r'\b\d+\b', ' digit ', text)
    return text


def train_model(data):
    if len(data) < 2:
        raise ValueError("Добавьте как минимум два текстовых примера для обучения модели.")

    x = [d['text'] for d in data]
    y = [d['tag'] for d in data]

    if len(set(y)) < 2:
        raise ValueError("Добавьте как минимум два различных тега для обучения модели.")

    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier(loss='hinge')),
    ])
    text_clf.fit(x, y)
    return text_clf


def predict_tag(question, model):
    return model.predict([question])[0]


def load_data():
    with open('model.json', 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data


def load_tags():
    with open('tags.json', 'r', encoding='utf-8') as file:
        tags = json.load(file)
    return tags


def save_data(data):
    with open('model.json', 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    tags = load_tags()
    if request.method == 'POST':
        text = request.form['text']
        tag = request.form['tag']
        try:
            data = load_data()
        except json.JSONDecodeError:
            data = []
        data.append({"text": text, "tag": tag})
        save_data(data)
        return redirect(url_for('add_data'))

    return render_template('add_data.html', message='Data added successfully!', tags=tags)


@app.route('/train_model', methods=['GET', 'POST'])
def train_model_route():
    model_data = load_data()
    data = load_data()
    num_items = len(data)
    items_per_page = 6
    num_pages = math.ceil(num_items // items_per_page)
    if request.method == 'POST':
        try:
            model = train_model(data)
            dump(model, 'trained_model.joblib')
            app.current_model = 'trained_model.joblib'
            return render_template('train_model.html', model_data=model_data, num_pages=num_pages)
        except ValueError as e:
            return render_template('train_model.html', error=str(e), message='Model trained unsuccessfully!',
                                   num_pages=num_pages)
        except Exception as ex:
            return render_template('train_model.html', error=str(ex),
                                   message='Error occurred during model training and saving!')

    return render_template('train_model.html', model_data=model_data, num_pages=num_pages)


@app.route('/test_model', methods=['GET', 'POST'])
def test_model():
    tags = load_tags()
    if request.method == 'POST':
        question = request.form['question']

        model_path = app.current_model
        if model_path:
            model = load(model_path)
            predicted_tag = predict_tag(question, model)
            return render_template('test_model.html', prediction=predicted_tag, tags=tags)
        else:
            return render_template('model_not_load.html')

    elif request.method == 'GET':
        return render_template('test_model.html')


@app.route('/answer_question', methods=['POST'])
def answer_question():
    question = request.form['question']
    predicted_tag = request.form['predicted_tag']
    correct_tag = request.form['correct_tag']
    correct = request.form.get('correct')

    if correct == 'true':
        tag = predicted_tag
    else:
        tag = correct_tag if correct_tag else "Неизвестный запрос"

    data = load_data()
    data.append({"text": question, "tag": tag})
    save_data(data)

    return render_template('test_model.html', prediction=None, model_data=load_data())


@app.route('/delete_question', methods=['POST'])
def delete_question():
    try:
        data = load_data()
        text_to_delete = request.form['text']
        data = [item for item in data if item['text'] != text_to_delete]
        save_data(data)
        return redirect(url_for('train_model_route'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

