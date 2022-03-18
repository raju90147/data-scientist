# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:12:50 2022
Name: _RAJU BOTTA____________ 
Batch ID: _05102021__________
"""


import pandas as pd

# import Dataset 

video_game = pd.read_csv("D:\Data Set\game.csv", encoding = 'utf8')
video_game.shape                       # shape of given dataset
video_game.columns
video_game.game                     # genre columns

from sklearn.feature_extraction.text import TfidfVectorizer 

#term frequency inverse document  frequency is a numerical statistic that is intended to reflect how important a word is to document in a collection or corpus

# Creating a Tfidf Vectorizer to remove all stop words

tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string

video_game["game"].isnull().sum() 
video_game["game"] = video_game["game"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(video_game.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape         # (5000, 3068)

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the  uclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity – metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of anime name to index number 

video_game_index = pd.Series(video_game.index, index = video_game['game']).drop_duplicates()

video_game_id = video_game_index["Resident Evil 4"]
video_game_id

def get_recommendations(game, topN):    
    
    # Getting the game index using its game 
    video_game_id = video_game_index[game]
    
    # Getting the pair wise similarity score for all the anime's with that 
    
    cosine_scores = list(enumerate(cosine_sim_matrix[video_game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    video_game_idx  =  [i[0] for i in cosine_scores_N]
    video_game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    video_game_similar_show = pd.DataFrame(columns=["game", "Score"])
    video_game_similar_show["game"] = video_game.loc[video_game_idx, "game"]
    video_game_similar_show["Score"] = video_game_scores
    video_game_similar_show.reset_index(inplace = True)  
    
    print (video_game_similar_show)

    
# Enter your game and number of video games to be recommended 

get_recommendations('Super Mario Galaxy 2', topN = 10)
video_game_index["The Legend of Zelda: Breath of the Wild"]



# =================== ********************* ======================== 
'''Problem Statement: -

The Entertainment Company, which is an online movie watching platform, wants to improve its collection of movies and showcase those that are highly rated and recommend those movies to its customer by their movie watching footprint. For this, the company has collected the data and shared it with you to provide some analytical insights and also to come up with a recommendation algorithm so that it can automate its process for effective recommendations. The ratings are between -9 and +9.
'''

#  Objective: To recommend movies to its customer by their movie watching footprint

import pandas as pd

#import dataset

enter = pd.read_csv('D:\Data Set\Entertainment.csv')    
enter.shape
enter.columns
enter.Category    #genre colomns

#import tfidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
    
#creating a tfidf vectorizer to remove stopwords

tfidf = TfidfVectorizer(stop_words = 'english') #taking stopwords from tfidf vectorizer

#replacing NaN values in overview column with empty string.

enter['Category'].isnull().sum()
enter['Category'] = enter['Category'].fillna(' ')

#preparing the tfidf matrix by fitting & transforming

tfidf_matrix = tfidf.fit_transform(enter.Category)

#transform a count matrix to a normalised tf or tfidf representation

tfidf_matrix.shape # (51,34)

# with the above matrix we need to find the similarity score
# there are several metrics for this such as the euclidean
# the pearson and the cosine similarity scores
# for now we will be using cosine similarity matrix
# a numeric quantity to represent the similarity b/w 2 movies
#cosine similarity metric is independent of magnitude and easy to calculate
# cosine(x,y) = (X.YT)/(||X||.||Y||)



from sklearn.metrics.pairwise import linear_kernel

#computing the cosine similarity on Tfidf matrix

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

#crating a mapping of entertainment name to index number

enter_index = pd.Series(enter.index, index = enter['Titles']).drop_duplicates()
enter_id = enter_index['GoldenEye (1995)']
enter_id

def get_recommendation(Name, topN): 
    
    #Getting movies index using its title
    
    enter_id = enter_index[Name]
    
    #getting the pair wise similarity score for all 
    
    cosine_scores = list(enumerate(cosine_sim_matrix[enter_id]))
    
    
    #sorting the cosine similarity scores based on scores
    
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse=True)
    
    #get the scores of top N most similar movies
    
    cosine_scores_N = cosine_scores[0:topN+1]
    
    #getting the movie index
    
    enter_idx = [i[0] for i in cosine_scores_N]
    enter_scores = [i[1] for i in cosine_scores_N]
    
    #similar movies & scores
    
    enter_similar_show = pd.DataFrame(columns=['Titles', 'Scores'])
    enter_similar_show['Score'] = enter.loc[enter_idx, 'Titles' ]
    enter_similar_show["Scores"]= enter_scores
    enter_similar_show.reset_index(inplace = True)
    print(enter_similar_show)
 
#recommending the movies using above function

get_recommendation("Sabrina (1995)", topN=10)

