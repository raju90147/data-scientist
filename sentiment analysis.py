# -*- coding: utf-8 -*-
"""
Name: _RAJU BOTTA____________ Batch ID: 05102021___________
Topic: Text Mining and NLP
    
"""
'''
Problem Statement: -
In the era of widespread internet use, it is necessary for businesses to understand what the consumers think of their products. If they can understand what the consumers like or dislike about their products, they can improve them and thereby increase their profits by keeping their customers happy. For this reason, they analyze the reviews of their products on websites such as Amazon or Snapdeal by using text mining and sentiment analysis techniques. 

Task 1:
1.	Extract reviews of any product from e-commerce website Amazon.
2.	Perform sentiment analysis on this extracted data and build a unigram and bigram word cloud. 
'''
# Solution: 1.

import requests
from bs4 import BeautifulSoup as bs #Beautiful soup is for web scrapping used to scrap specific content
import re 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# web scraping

lenovo_laptop_reviews = []
for i in range(1,11):
    ip = []
    url = "https://www.amazon.in/Lenovo-Ideapad-Laptop-Windows-81MV00WRIN/product-reviews/B07VX52JD2/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber="+str(i)
    response = requests.get(url)
soup = bs(response.content, 'html.parser') #creating soup object to iterate over the extracted content
reviews = soup.find_all('span', attrs={"class", "a size base review text content"})

for i in range(len(reviews)):
    ip.append(reviews[i].text)
lenovo_laptop_reviews = lenovo_laptop_reviews+ip  #adding the reviews of one page to empty list which in future contains all the reviews

# Wrting reviews in a text file  
with open("lenovo_laptop.txt",'w',encoding = 'utf8') as output:
    output.write(str(lenovo_laptop_reviews))
    
   # Joinding all reviews into single paragraph
ip_rev_string = " " .join(lenovo_laptop_reviews)

import nltk 
from nltk.corpus import stopwords   

# removing unwanted symbols incase if exists

ip_rev_string = re.sub("[^A-Z a-z ""]+","",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9""]+","",ip_rev_string)

# words that contained in lenovo ideapad laptop reviews 

ip_reviews_words = ip_rev_string.split(" ")

#TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(ip_reviews_words)

with open(r"C:\Users\rajub\lenovo_laptop.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")

stop_words.extend(["lenovo","laptop","time","ideapad","device","screen","battery","product","good","day","price"])
other_stopwords_to_remove = ['\\n', 'n', '\\', '>', 'nLines', 'nI',"n'", "hi"]

stop_words = stop_words.union(set(other_stopwords_to_remove))

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

stop_words = stop_words.union(set(other_stopwords_to_remove))
stop_words = set(stop_words)
text = str(stop_words)

wordcloud = WordCloud(width = 1800, height = 1800, 
                background_color ='white', 
                max_words=200,
                stopwords = stopwords, 
                min_font_size = 10).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# positive words # Choose the path for +ve words stored in system
with open("D:\datasets\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# negative words Choose path for -ve words stored in system
with open("D:\datasets\negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['price', 'great'] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud

words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# ====================== ********************* ============================
'''
Task 2:
1.	Extract reviews for any movie from IMDB and perform sentiment analysis. '''

import requests
from bs4 import BeautifulSoup as bs #Beautiful soup is for web scrapping used to scrap specific content
import re 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# web scraping

Jai_bhim_movie_reviews = []
for i in range(1,11):
    ip = []
    url = " https://www.imdb.com/title/tt15097216/reviews/?ref_=tt_ql_urv"+str(i)
    response = requests.get(url)
soup = bs(response.content, 'html.parser') #creating soup object to iterate over the extracted content
reviews = soup.find_all('span', attrs={"class", "lister-item-content"})

for i in range(len(reviews)):
    ip.append(reviews[i].text)
Jai_bhim_movie_reviews = Jai_bhim_movie_reviews +ip  #adding the reviews of one page to empty list which in future contains all the reviews

# Wrting reviews in a text file  

with open("jaibhim_reviews.txt",'w',encoding = 'utf8') as output:
    output.write(str(Jai_bhim_movie_reviews))
    
   # Joinding all reviews into single paragraph

ip_rev_string = " " .join(Jai_bhim_movie_reviews)

import nltk 
from nltk.corpus import stopwords   

# removing unwanted symbols incase if exists

ip_rev_string = re.sub("[^A-Z a-z ""]+","",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9""]+","",ip_rev_string)

# words that contained in lenovo ideapad laptop reviews 

ip_reviews_words = ip_rev_string.split(" ")

#TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(ip_reviews_words)

with open(r"C:\Users\rajub\jaibhim_movie.txt","r") as sw:
    stop_words = sw.read()

#stop_words = stop_words.split("\n")

#other_stopwords_to_remove = ['\\n', 'n', '\\', '>', 'nLines', 'nI',"n'", "hi"]

#stop_words = stop_words.union(set(other_stopwords_to_remove))

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

#stop_words = stop_words.union(set(other_stopwords_to_remove))
stop_words = set(stop_words)
text = str(stop_words)

wordcloud = WordCloud(width = 1800, height = 1800, 
                background_color ='white', 
                max_words=200,
                stopwords = stopwords, 
                min_font_size = 10).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# positive words # Choose the path for +ve words stored in system

with open("D:\datasets \Data\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# negative words Choose path for -ve words stored in system
with open("D:\datasets \Data\\negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)

# Remove stop words
text_content = [word for word in text_content if word not in stopwords_wc]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud

words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords= stopwords_wc)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# ========================== ********************* ======================
''' 
Task 3: 
1.	Choose any other website on the internet and do some research on how to extract text and perform sentiment analysis
'''

import requests
from bs4 import BeautifulSoup as bs #Beautiful soup is for web scrapping used to scrap specific content
import re 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# web scraping

vivo_phone_reviews = []
for i in range(1,11):
    ip = []
    url = " https://www.flipkart.com/vivo-y20-dawn-white-64-gb/p/itm10285cb0a8dd9"+str(i)
    response = requests.get(url)
soup = bs(response.content, 'html.parser') #creating soup object to iterate over the extracted content
reviews = soup.find_all('span', attrs={"class", "_2QKOHZ"})

for i in range(len(reviews)):
    ip.append(reviews[i].text)
vivo_phone_reviews = vivo_phone_reviews +ip   #adding the reviews of one page to empty list which in future contains all the reviews

# Wrting reviews in a text file  

with open("vivo_phone_reviews.txt",'w',encoding = 'utf8') as output:
    output.write(str(vivo_phone_reviews))
    
   # Joinding all reviews into single paragraph

ip_rev_string = " " .join(vivo_phone_reviews)

import nltk 
from nltk.corpus import stopwords   

# removing unwanted symbols incase if exists

ip_rev_string = re.sub("[^A-Z a-z ""]+","",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9""]+","",ip_rev_string)

# words that contained in lenovo ideapad laptop reviews 

ip_reviews_words = ip_rev_string.split(" ")

#TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(ip_reviews_words)

with open(r"C:\Users\rajub\vivophone_reviews.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")

other_stopwords_to_remove = ['\\n', 'n', '\\', '>', 'nLines', 'nI',"n'", "hi"]

#stop_words = stop_words.union(set(other_stopwords_to_remove))

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

#stop_words = stop_words.union(set(other_stopwords_to_remove))
stop_words = set(stop_words)
text = str(stop_words)

wordcloud = WordCloud(width = 1800, height = 1800, 
                background_color ='white', 
                max_words=200,
                stopwords = stopwords, 
                min_font_size = 10).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# positive words # Choose the path for +ve words stored in system

with open("D:\datasets \Data\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# negative words Choose path for -ve words stored in system
with open("D:\datasets \Data\\negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)

# Remove stop words
text_content = [word for word in text_content if word not in stopwords_wc]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud

words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords= stopwords_wc)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()




