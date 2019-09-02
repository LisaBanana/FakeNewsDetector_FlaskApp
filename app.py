# largely inspired by this article : https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)
# load model
NB_FakeNews_model = open('NB_FakeNews_model.pkl','rb')
clf = joblib.load(NB_FakeNews_model)
# load vectorizer
cvect = open('Vocab.pickle','rb')
cv = joblib.load(cvect)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	#df=pd.read_csv('news_dataset.csv', index_col = 0)
	#df['label'] = df['label'].map({'real': 0, 'fake': 1})
	#df = df.fillna(' ')
	#X = df['title']
	#y = df['label']

	#cv = CountVectorizer()
	#X = cv.fit_transform(X)
	#
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	#Naive Bayes Classifier (others where tested go to the notebook to see accuracy on train and test results)

	#clf = MultinomialNB()
	#clf.fit(X_train,y_train)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)