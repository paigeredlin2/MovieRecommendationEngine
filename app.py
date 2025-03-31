from flask import Flask, request
from flask import render_template
from livereload import Server
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from imdb import Cinemagoer
import numpy as np
import pandas as pd
import concurrent.futures


df = pd.read_csv('movies.csv')
df = df.sort_values(by=['imdbId'])
ia = Cinemagoer()

vectorizer = TfidfVectorizer(ngram_range=(1,2))
tf_title = vectorizer.fit_transform(df['title'])

def add_covers(imdb_id):
    imdb_id = int(imdb_id)
    movie = ia.get_movie(imdb_id)
    return movie['cover url']

def add_rating(imdb_id):
    imdb_id = int(imdb_id)
    movie = ia.get_movie(imdb_id)
    return movie['rating']

def concurrent_cover(mini_frame):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        covers = list(executor.map(add_covers, mini_frame['imdbId']))
        return covers

def concurrent_rating(mini_frame):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        rating = list(executor.map(add_rating, mini_frame['imdbId']))
        return rating

def jaccard_similarity(base_case_genres: str, compartor_genres: str):
    compartor_genres = compartor_genres.split('|')
    base_case_genres = base_case_genres.split('|')

    genres_categories = {'total': 0}
    for genre in base_case_genres:
        if genre in genres_categories:
            genres_categories[genre] += 1
        else:
            genres_categories[genre] = 1
        genres_categories['total'] += 1

    denominator = genres_categories['total']
    numerator = 0
    for genre in compartor_genres:
        if genre in genres_categories:
            numerator += genres_categories[genre]

    return float(numerator) / float(denominator)

def cosine_similarity_func(query, query_vec, min_similar):
    similarity = cosine_similarity(query_vec, tf_title).flatten()
    min_similarity = min_similar

    indices = np.where(similarity >= min_similarity)

    similar_movies = df.iloc[indices].copy()
    similar_movies[f'similarity{query}'] = similarity[indices]
    similar_movies = similar_movies[similar_movies[f'similarity{query}'] < 1]
    similar_movies = similar_movies.sort_values(by=f'similarity{query}', ascending=False)
    return similar_movies

def find_similar(previous_movies):
    k = 10
    combined_genres = "|".join(movie['genres'] for movie in previous_movies)
    comparison_type = "genres"
    df['jaccard'] = df[comparison_type].map(lambda x: jaccard_similarity(combined_genres, x))

    similar_genres = df.sort_values(by='jaccard', ascending=False)

    previous_movies_ids = [movie['movieId'] for movie in previous_movies]
    similar_genres = similar_genres[~similar_genres['movieId'].isin(previous_movies_ids)]

    titles_list = list(movie['title'] for movie in previous_movies)

    similar_titles = []
    for title in titles_list:
        title_vec = vectorizer.transform([title])
        title_similar = cosine_similarity_func(title, title_vec, 0.45)
        similar_titles.append(title_similar)

    if similar_titles:
        similar_titles_df = pd.concat(similar_titles)
        similar_movies = pd.concat([similar_titles_df, similar_genres.head(k)])
    else:
        similar_movies = similar_genres.head(k)

    return similar_movies.head(k)



app = Flask(__name__)
app.config["DEBUG"] = True

previous_movies = []

@app.route('/')
def index():
    movies = df.sample(n=10)
    movies['covers'] = concurrent_cover(movies)
    movies['rating'] = concurrent_rating(movies)

    return render_template("index.html", movies=movies)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('searchbox')
    query_vec = vectorizer.transform([query])
    similar_movies = cosine_similarity_func(query, query_vec, 0.3)

    similar_movies['covers'] = concurrent_cover(similar_movies)
    similar_movies['rating'] = concurrent_rating(similar_movies)

    return render_template("index.html", movies=similar_movies.head(10))


@app.route('/recommend', methods=['GET'])
def recommend():
    #if user selects a movie
    selected_movie = request.args.get('movie_id')

    #if user clicks reset_search, go back to previous movie recommendations
    reset_search = request.args.get('reset_search')
    if reset_search:
        rereturn_similar_movies = find_similar(previous_movies)
        rereturn_similar_movies['covers'] = concurrent_cover(rereturn_similar_movies)
        rereturn_similar_movies['rating'] = concurrent_rating(rereturn_similar_movies)
        return render_template("index.html", movies=rereturn_similar_movies, previous_movies=previous_movies)

    #if user deletes a movie from their previous recommendation
    delete_movie = request.args.get('del_movie_id')
    if delete_movie:
        previous_movies[:] = [movie for movie in previous_movies if movie['movieId'] != int(delete_movie)]
        if not previous_movies:
            movies = df.sample(n=10)
            movies['covers'] = concurrent_cover(movies)
            movies['rating'] = concurrent_rating(movies)
            return render_template("index.html", movies=movies)

    if not selected_movie:
        ten_similar_after_del = find_similar(previous_movies)
        ten_similar_after_del['covers'] = concurrent_cover(ten_similar_after_del)
        ten_similar_after_del['rating'] = concurrent_rating(ten_similar_after_del)
        return render_template("index.html", movies=ten_similar_after_del, previous_movies=previous_movies)
    else:
        base_case = df.loc[df['movieId'] == int(selected_movie)].squeeze()

    base_case = base_case.to_dict()
    previous_movies.append(base_case)


    ten_similar_movies = find_similar(previous_movies)
    ten_similar_movies['covers'] = concurrent_cover(ten_similar_movies)
    ten_similar_movies['rating'] = concurrent_rating(ten_similar_movies)

    return render_template("index.html", movies=ten_similar_movies, selected_movie=base_case, previous_movies=previous_movies)


if __name__ == "__main__":
    server = Server(app.wsgi_app)
    server.watch("static/*")  # Watch template files for changes
    server.watch("templates/*")  # Watch template files for changes
    server.serve(port=5000)  # Start the server with live reload