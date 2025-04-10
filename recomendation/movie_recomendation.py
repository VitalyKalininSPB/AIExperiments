# data processing

import numpy as np
import pandas as pd
import scipy.stats

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# similarity
from sklearn.metrics.pairwise import cosine_similarity

# import ratings
ratings_df = pd.read_csv("ml-latest-small/ratings.csv")

# import movies
movies_df = pd.read_csv("ml-latest-small/movies.csv")

ratings_df.head()
movies_df.head()

print(ratings_df.info())
print(movies_df.info())

# Number of unique users
print("The ratings dataset has", ratings_df["userId"].nunique(), "unique users")
# Number of movies
print("The ratings dataset has", ratings_df["movieId"].nunique(), "unique movies")
# Number of unique ratings
print("The ratings dataset has", ratings_df["rating"].nunique(), "unique ratings")
# List of unique ratings
print("The unique ratings are", sorted(ratings_df["rating"].unique()))
full_df = pd.merge(ratings_df, movies_df, on="movieId", how="inner")
mean_ratings = full_df.groupby("title")["rating"].mean()
number_of_ratings = full_df.groupby("title")["rating"].count()
full_df["title"].value_counts()

sns.jointplot(x=mean_ratings, y=number_of_ratings)
plt.show()
full_df["title"].value_counts()

filtered_df = full_df[full_df.groupby("movieId")["movieId"].transform("size") > 30]
filtered_df_final = filtered_df[filtered_df.groupby("userId")
                                ["userId"].transform("size") > 20]

user_movie_matrix = filtered_df_final.pivot_table(index="userId", columns="title", values="rating")
matrix_norm = user_movie_matrix.subtract(user_movie_matrix.mean(axis=1), axis = 0)
matrix_filled = matrix_norm.fillna(0)

user_simi_cosine = cosine_similarity(matrix_filled)
user_simi_df = pd.DataFrame(user_simi_cosine, columns=matrix_filled.index.values,
                           index=matrix_filled.index.values)

# Pick a user ID
picked_userid = 7

# Remove picked user ID from the candidate list
user_simi_df.drop(index=picked_userid, inplace=True)
# Number of similar users
n = 5

# User similarity threshold
user_similarity_threshold = 0.1

# Get top n similar users
similar_users = user_simi_df[user_simi_df[picked_userid]>user_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]
# Print out top n similar users
print(f'The similar users for user {picked_userid} are', similar_users)

# Movies that the target user has watched
picked_userid_watched = user_movie_matrix[user_movie_matrix.index == picked_userid].dropna(axis=1, how='all')

# Movies that similar users watched. Remove movies that none of the similar users have watched
similar_user_movies = user_movie_matrix[user_movie_matrix.index.isin(similar_users.index)].dropna(axis=1, how='all')

# Remove the watched movie from the movie list
similar_user_movies.drop(picked_userid_watched.columns, axis=1, inplace=True, errors='ignore')

# A dictionary to store item scores
item_score = {}
# Loop through items
for i in similar_user_movies.columns:
  # Get the ratings for movie i
  movie_rating = similar_user_movies[i]
  # Create a variable to store the score
  total = 0
  # Create a variable to store the number of scores
  count = 0
  # Loop through similar users
  for u in similar_users.index:
    # If the movie has rating
    if pd.isna(movie_rating[u]) == False:
      # Score is the sum of user similarity score multiply by the movie rating
      score = similar_users[u] * movie_rating[u]
      # Add the score to the total score for the movie so far
      total += score
      # Add 1 to the count
      count +=1
  # Get the average score for the item
  item_score[i] = total / count
# Convert dictionary to pandas dataframe
item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])

# Sort the movies by score
ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)

# Select top m movies
m = 5
print(ranked_item_score.head(m))
