import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Dot, Concatenate
from tensorflow.keras.models import Model

import kagglehub

# Download latest version
path = kagglehub.dataset_download("sahilkirpekar/goodreads10k-dataset-cleaned")

print("Path to dataset files:", path)

# ratings dataset
ratings_df = pd.read_csv(path + "/Ratings.csv")

# books dataset
books_df = pd.read_csv(path + "/Books.csv")

print(ratings_df.head())
print(books_df.head())

# unique books and users
n_users = ratings_df.user_id.nunique()
n_books = ratings_df.book_id.nunique()

print("The number of users - ", n_users)
print("The number of books - ", n_books)

train_df, test_df = train_test_split(ratings_df, test_size=0.3, random_state=13)

# create book embedding path
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)

# creating user embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# concatenate features
conc = Concatenate()([book_vec, user_vec])

# add fully connected layers
fc1 = Dense(128, activation="relu")(conc)
fc2 = Dense(32, activation="relu")(fc1)
out = Dense(1)(fc2)

# Create a model and compile it
model = Model([user_input, book_input], out)
model.compile("adam", "mean_squared_error")
print(model.summary())

#history = model.fit(x=[train_df.user_id, train_df.book_id], y=train_df.rating, epochs=10, verbose=1)
#model.evaluate([test_df.user_id, test_df.book_id], test_df.rating)

# predictions
predictions = model.predict([test_df.user_id.head(10), test_df.book_id.head(10)])
for i in range(0,10):
    print("Predicted Rating - ", predictions[i], "Actual Rating - ", test_df.rating.iloc[i])

# create dataset for making recommendations for the picked user
picked_userid = 150
book_data = np.array(list(set(test_df.book_id)))
user = np.array([picked_userid for i in range(len(book_data))])
print(len(user))
print(len(book_data))


dataset = [user, book_data]

predict_1 = model.predict(dataset)
predictions = np.array([a[0] for a in predict_1])
recommended_book_ids = (-predictions).argsort()[:5]
print(recommended_book_ids)
# print predicted scores
print(predictions[recommended_book_ids])

# recommended books for picked user
books_df[books_df["book_id"].isin(recommended_book_ids)]
