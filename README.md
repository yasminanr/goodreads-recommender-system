# Book Recommender System
What we need:
- Data on book ratings
- Meta-data of books
- List of books that we like

We will use book and rating data from Goodreads scraped and collected by researchers at UCSD.

You can get the data [here](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).
<br>
Read the papers [here](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/recsys18_mwan.pdf) and [here](https://aclanthology.org/P19-1248/).

There are 3 files that we will use:
- "goodreads_books.json.gz", contains the book meta-data.
- "goodreads_interactions.csv", contains the data on users and ratings.
- "book_id_map.csv", to match the book ID on the 2 files above.

Project steps:
- Build a simple book search engine.
- Create our own list of books that we like (mine has 20 books that I love, you can see it in the file "my_liked_books.csv").
- Build a recommender system using user-based collaborative filtering.
