import pandas as pd
import numpy as np

# Load the dataset into a pandas dataframe
df = pd.read_csv(r'D:\Templates\UW Stuff\Classes\MSBA\Classes\PM Stuff for me\Environments\movies_metadata.csv',
                 low_memory=False)
# Display the first five movies in the dataframe
#df.head()


#Calculate the number of votes garnered by the 80th percentile movie
#m

#Only consider movies longer than 45 minutes and shorter than 300 minutes
#q_movies = df[(df['runtime'] >= 45) & (df['runtime'] <= 300)]
#Only consider movies that have garnered more than m votes
#q_movies = q_movies[q_movies['vote_count'] >= m]
#Inspect the number of movies that made the cut
#q_movies.shape

# Calculate C
#C = df['vote_average'].mean()
#C

# Function to compute the IMDB weighted rating for each movie
#def weighted_rating(x, m=m, C=C):
   # v = x['vote_count']
   # R = x['vote_average']
# Compute the weighted score
    #return (v/(v+m) * R) + (m/(m+v) * C)

# Compute the score using the weighted_rating function defined above
#q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#q_movies.sort_values('score')

df.columns

#Only keep those features that we require
df = df[['title','genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]
df.head()

#Convert release_date into pandas datetime format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
#Extract year from the datetime
df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

#Helper function to convert NaT to 0 and all other years to integers.
def convert_int(x):
    try:
        return int(x)
    except:
        return 0

#Apply convert_int to the year feature
df['year'] = df['year'].apply(convert_int)

#Drop the release_date column
df = df.drop('release_date', axis=1)
#Display the dataframe
df.head()

#Print genres of the first movie
df.iloc[0]['genres']

#Import the literal_eval function from ast
from ast import literal_eval

#Define a stringified list and output its type
a = "[1,2,3]"
print(type(a))
#Apply literal_eval and output type
b = literal_eval(a)
print(type(b))

#Convert all NaN into stringified empty lists
df['genres'] = df['genres'].fillna('[]')
#Apply literal_eval to convert to the list object
df['genres'] = df['genres'].apply(literal_eval)
#Convert list of dictionaries to a list of strings
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df.head()

#Create a new feature by exploding genres
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
#Name the new feature as 'genre'
s.name = 'genre'
#Create a new dataframe gen_df which by dropping the old 'genres' feature and adding the new 'genre'.
gen_df = df.drop('genres', axis=1).join(s)
#Print the head of the new gen_df
gen_df.head()