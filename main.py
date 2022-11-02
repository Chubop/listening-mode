# https://source.unsplash.com/1920x1080/?cow

from flask import Flask, request
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
model = AutoModelForTokenClassification.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")

pipeline_ = pipeline('ner', model=model, tokenizer=tokenizer)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/vi", methods=['POST'])
def voice_input():

    content = request.json
    if 'tokens' not in content:
        return None # TODO error casing
    entity = pipeline_(content['tokens'])
    if len(content['tokens']) > 0:
        res = {'keyword': "https://source.unsplash.com/1920x1080/?" + (entity[0]['word'])}
    else:
        res = {'keyword': None}
    print('Unsplash URL:', res['keyword'])
    return res


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)
