# -*- codingL utf-8 -*-
from flask import Flask, render_template, request

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.tokenize import TreebankWordTokenizer

app = Flask(__name__)

loaded_model = load_model('Flask-Web-Server/model/best_model.h5')

def sentiment_predict(new_sentence):
  new_sentence = TreebankWordTokenizer().tokenize(new_sentence)#토큰화
  #new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = Tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  print(encoded)
  pad_new = pad_sequences(encoded, maxlen = 42) # 패딩
  #print(pad_new)
  score = float(loaded_model.predict(pad_new)) # 예측
  print(score)
  if(score > 0.5):
    return "옳은 정보입니다."
  else:
    return "잘못된 정보입니다."

@app.route("/", methods=['GET','POST'])
def index():
	if request.methods == 'GET':
		return render_template('index.html')
	if request.methods == 'POST':
		sentence = request.form['inputext']
		result = sentiment_predict(sentence)

		return render_template('index.html', result = result)
if __name__ == '__main__':
	app.run(debug=True)
