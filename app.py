# import library
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

df = pd.read_excel('datasets\Master Data Knowledge Management (AI) HRIS.xlsx', header=1)
df.drop(columns=['NO'], inplace=True)
df['Division'].fillna(method='ffill', inplace=True)
df['Project Title'].fillna(method='ffill', inplace=True)
df['Location'].fillna(method='ffill', inplace=True)
df['Department'].fillna(method='ffill', inplace=True)
df['Location'] = df['Location'].str.capitalize()

df['Improve'] = df['Improve'].str.replace(r'^\d+\.', '', regex=True)
df

kota = pd.unique(df['Location'])
kota

# Preprocessing
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()] 
    tokens = [word for word in tokens if word not in stopwords]   
    return ' '.join(tokens)

df['Cleaned_Title'] = df['Project Title'].apply(preprocess_text)
df['Cleaned_Location'] = df['Location'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Title'] + ' ' + df['Cleaned_Location'])

def get_recommendations(project_title, location, top_n=5):
    query = preprocess_text(project_title + ' ' + location)
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    top_indices = related_docs_indices[:top_n]
    recommendations = df.iloc[top_indices]['Improve'].tolist()
    return recommendations

@app.route('/', methods=['GET'])
def index():
    return render_template('addCI.html')

@app.route('/test', methods=['POST'])
def test():
    project_title = request.form['project_title']
    location = ''
    title_fix = ''

    title_split = project_title.split()

    for prompt in title_split:
        if prompt.capitalize() in kota:
            location = prompt
        else: 
            title_fix = prompt

    recommendations = get_recommendations(title_fix, location)

    response = {"title_fix": title_fix, "location": location, "recommendations": recommendations, "message": "Data berhasil diterima oleh server"}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
