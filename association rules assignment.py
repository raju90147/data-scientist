# -*- coding: utf-8 -*-
"""
author@ raju botta
Association rules"""

# conda install mlxtend
# or
# pip install mlxtend

# book dataset

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
book = []
with open("/content/book.csv") as f:
    book = f.read()

# splitting the data into separate transactions using separator as "\n"
book = book.split("\n")


books_list = []
for i in book:
    books_list.append(i.split(","))

all_books_list = [i for item in books_list for i in item]


from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_books_list)


# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
books_series = pd.DataFrame(pd.Series(books_list))
books_series = books_series.iloc[:11, :] # removing the last empty transaction

books_series.columns = ["transactions"]


# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = books_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.05, max_len = 4, use_colnames = True)


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

 

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head()
rules.sort_values('lift', ascending = False).head(10)

 
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

 # ============ ********************* *************************
# Grocery dataset


grocery = []
with open("/content/groceries.csv") as f:
    grocery = f.read()

# splitting the data into separate transactions using separator as "\n"
grocery = grocery.split("\n")

grocery_list = []
for i in grocery:
    grocery_list.append(i.split(","))

all_grocery_list = [i for item in grocery_list for i in item]


from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_grocery_list)


# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

 

# Creating Data Frame for the transactions data
grocery_series = pd.DataFrame(pd.Series(grocery_list))
grocery_series = grocery_series.iloc[:9835, :] # removing the last empty transaction

grocery_series.columns = ["transactions"]


# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = grocery_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.05, max_len = 4, use_colnames = True)


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()
 

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)
 

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

 # ================= *************** =================================
# mymovies dataset

movies = []
with open("/content/my_movies.csv") as f:
    movies = f.read()

# splitting the data into separate transactions using separator as "\n"
movies = movies.split("\n")

movies_list = []
for i in movies:
    movies_list.append(i.split(","))

all_movies_list = [i for item in movies_list for i in item]

from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_movies_list)


# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:16], x = list(range(0, 16)), color = 'rgbkymc')
plt.xticks(list(range(0, 16), ), items[0:16])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
movies_series = pd.DataFrame(pd.Series(movies_list))
movies_series = movies_series.iloc[:11, :] # removing the last empty transaction

movies_series.columns = ["transactions"]


# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = movies_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.05, max_len = 4, use_colnames = True)


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

 # ==================== ******************* ===================================
# my phone data

my_phone_data = []
with open("/content/myphonedata.csv") as f:
    my_phone_data = f.read()

# splitting the data into separate transactions using separator as "\n"
my_phone_data = my_phone_data.split("\n")

my_phone_data_list = []
for i in my_phone_data:
    my_phone_data_list.append(i.split(","))

all_my_phone_data_list = [i for item in my_phone_data_list for i in item]

from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_my_phone_data_list)


# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:10], x = list(range(0, 10)), color = 'rgbkymc')
plt.xticks(list(range(0, 10), ), items[0:10])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
phone_data_series = pd.DataFrame(pd.Series(my_phone_data_list))
phone_data_series = phone_data_series.iloc[:11, :] # removing the last empty transaction

phone_data_series.columns = ["transactions"]


# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = phone_data_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.05, max_len = 4, use_colnames = True)


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

# =================== ******************** ========================
# transaction retail

transaction = []
with open("/content/transactions_retail1.csv") as f:
    transaction = f.read()

# splitting the data into separate transactions using separator as "\n"
transaction = transaction.split("\n")


transaction_retail = []
for i in transaction:
    transaction_retail.append(i.split(","))

all_transaction_retail = [i for item in transaction_retail for i in item]


from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_transaction_retail)


# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:7], x = list(range(0, 7)), color = 'rgbkymc')
plt.xticks(list(range(0, 7), ), items[0:7])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
transaction_retail_series = pd.DataFrame(pd.Series(transaction_retail))
transaction_retail_series = transaction_retail_series.iloc[:11, :] # removing the last empty transaction

transaction_retail_series.columns = ["transactions"]


# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = transaction_retail_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.05, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 7)), height = frequent_itemsets.support[0:7], color ='rgmyk')
plt.xticks(list(range(0, 7)), frequent_itemsets.itemsets[0:7], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)
