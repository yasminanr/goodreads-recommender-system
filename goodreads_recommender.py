import numpy as np
import pandas as pd

def make_clickable(val):
    return '<a target="_blank" href="{}">Goodreads</a>'.format(val)

def show_image(val):
    return '<img src="{}" width=50></img>'.format(val)

def init_data(liked_books: str, book_id_map: str, interactions: str, book_titles: str):
    my_books = pd.read_csv(liked_books, index_col=0)
    my_books["book_id"] = my_books["book_id"].astype(str)
    my_books["mod_title"] = my_books["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True).str.lower()
    my_books["mod_title"] = my_books["mod_title"].str.replace("\s+", " ", regex=True)

    csv_mapping = {}
    with open(book_id_map, "r") as f:
        for line in f:
            csv_id, book_id = line.strip().split(",")
            csv_mapping[csv_id] = book_id

    book_set = set(my_books["book_id"])
    overlap_users = {}
    with open(interactions, "r") as f:
        for line in f:
            user_id, csv_id, _, rating, _ = line.split(",")
            book_id = csv_mapping.get(csv_id)
            if book_id in book_set:
                if user_id not in overlap_users:
                    overlap_users[user_id] = 1
                else:
                    overlap_users[user_id] += 1
    filtered_overlap_users = set([k for k in overlap_users if overlap_users[k] > (my_books.shape[0]/5)])

    interactions_list = []
    with open(interactions, "r") as f:
        for line in f:
            user_id, csv_id, _, rating, _ = line.split(",")
            if user_id in filtered_overlap_users:
                book_id = csv_mapping[csv_id]
                interactions_list.append([user_id, book_id, rating])
    interactions = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"])
    interactions = pd.concat([my_books[["user_id", "book_id", "rating"]], interactions])
    interactions["book_id"] = interactions["book_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"])
    interactions["user_index"] = interactions["user_id"].astype("category").cat.codes
    interactions["book_index"] = interactions["book_id"].astype("category").cat.codes
    
    book_titles = pd.read_json(book_titles)
    book_titles["book_id"] = book_titles["book_id"].astype(str)
    
    return interactions, my_books, book_titles

def show_recommendation(interactions, my_books, book_titles):
    from scipy.sparse import coo_matrix
    ratings_mat_coo = coo_matrix((interactions["rating"], (interactions["user_index"], interactions["book_index"])))
    ratings_mat = ratings_mat_coo.tocsr()

    my_index = 0
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(ratings_mat[my_index,:], ratings_mat).flatten()
    indices = np.argpartition(similarity, -15)[-15:]
    similar_users = interactions[interactions["user_index"].isin(indices)].copy()
    similar_users = similar_users[similar_users["user_id"]!="-1"]

    book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean'])
    book_recs = book_recs.merge(book_titles, how="inner", on="book_id")
    
    book_recs["adjusted_count"] = book_recs["count"] * (book_recs["count"] / book_recs["ratings"])
    book_recs["score"] = book_recs["mean"] * book_recs["adjusted_count"]
    book_recs = book_recs[~book_recs["book_id"].isin(my_books["book_id"])]
    book_recs = book_recs[~book_recs["title_clean"].isin(my_books["mod_title"])]
    book_recs = book_recs[book_recs["mean"] >= 4]
    book_recs = book_recs[book_recs["count"] > 2]
    book_recs = book_recs[["book_id", "count", "mean", "title", "ratings", "url", "cover_image", "title_clean"]]
    top_recs = book_recs.sort_values("mean", ascending=False)
    return top_recs.head(10).style.format({'url': make_clickable, 'cover_image': show_image})
