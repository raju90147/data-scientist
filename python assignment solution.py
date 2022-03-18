# -*- coding: utf-8 -*-
"""
Name: __RAJU BOTTA___________ Batch ID: __05102021_________
Topic: Business Understanding

Python Module- 1 Assignments
ASSIGNMENT-1 Data Types

"""
	

1#a) concatenation of 2 lists

List1 = [10, 3.14, "raju", 5+6j, True]

List2 = [15, 2.10, "botta", 3+2j, False]

List3 = List1 + List2 #concatenation of 2 lists
print("concatenation of 2 lists", List3)

# output: concatenation of 2 lists [10, 3.14, 'raju', (5+6j), True, 15, 2.1, 'botta', (3+2j), False]
________________________________________

1#b)find frequency of each element in concatenated list
    
import collections # import collections package

con_list = [10, 3.14, "raju", 5+6j, True, 15, 2.10, "botta", 3+2j, False]

freq = collections.Counter(con_list)

print(dict(freq))

# output: {10: 1, 3.14: 1, 'raju': 1, (5+6j): 1, True: 1, 15: 1, 2.1: 1, 'botta': 1, (3+2j): 1, False: 1}
________________________________________


#or another way using iteration
con_list = [10, 3.14, "raju", 5+6j, True, 15, 2.10, "botta", 3+2j, False]

freq = {}  #empty dictionary 

#iterating over the list

for ele in con_list:
    if ele in freq:   #using list.count()
        freq[ele] = con_list.count(ele)

for key,value in freq.items(): 
    print(key,'-',value)   #print dictionary format (key,value pair)     

# output: {10: 1, 3.14: 1, ‘raju’: 1, (5+6j): 1, True: 1, 15: 1, 2.1: 1, ‘botta’: 1, (3+2j): 1, False: 1}
 	
 	    
1#c) print the list in reverse order

con_list = [10, 3.14, "raju",5+6j, True, 15, 2.10, "botta",3+2j, False]
    
con_list.reverse()
print("reverse list ",con_list)

# output : reverse list  [False, (3+2j), 'b‘tta',’2.1, 15, True, (5+6j), 'r‘ju',’3.14, 10]
________________________________________


#2 sets
       

set1 = {1,2,3,4,5,6,7,8,9,10}
set2 = {5,6,7,8,9,10,11,12,13,14,15}
dir(set)

#a) common elements in above 2 sets

set1.intersection(set2)     #using intersection

# out: {5, 6, 7, 8, 9, 10}

#b) elements that are not common
set1^set2 #using xor operator

# out: {1, 2, 3, 4, 11, 12, 13, 14, 15}

    
# or another way using difference
s1 = set1.difference(set2)
s2 = set2.difference(set1)    
print("elements that are not common are: {0}{1}"  .format(s1,s2))
    
#c) remove element 7 from both sets
print(set1.remove(7))
set1
print(set2.remove(7))    
set2

#3 create dictionary 

dict1 = {'AP':16000, 'Telangana':17000, 'Tamilnadu':21000, 'kerala':25000, 'Maharashtra':31000}

#a) print only state names
print(dict1.keys()) #print keys 

# out: dict_keys(['AP', 'Telangana', 'Tamilnadu', 'kerala', 'Maharashtra'])

#b) update with another country cases
dict1['delhi'] = 35000
print(dict1)

# out: {'AP': 16000, 'Telangana': 17000, 'Tamilnadu': 21000, 'kerala': 25000, 'Maharashtra': 31000, 'delhi': 35000}
________________________________________

#Module 2 - Operators
'''
Please implement by using Python
1.	A. Write an equation which relates   399, 543 and 12345 
B.  “When I divide 5 with 3, I got 1. But when I divide -5 with 3, I got -2”—How would you justify it.
       2.  a=5,b=3,c=10.. What will be the output of the following:
              A. a/=b
              B. c*=5  
       3. A. How to check the presence of an alphabet ‘s’ in the word “Data Science” .
            B. How can you obtain 64 by using numbers 4 and 3 .'''

#1)a) write an equation which relates 399, 543, and 12345

# 12345/543 has a quotient of 22 and yields a remainder of 399

x = 12345
y = 543
z = 399

equation = 22*y + z

if equation == x:
    print('it is a valid relation')

 # out : it is a valid relation
________________________________________

#2) a=5, b=3, c=10
#A) a/=b
a = 5
b = 3
c = 10

a/=b # a=a/b
print(a)

# out: 1.6666666666666667
________________________________________

#B) c*=5

c*=5 #c = c*5
print(c)

# out : 50

#3) A. How to check presence of an alphabet 's' in word "Data Science"

word = "Data Science"
alphabet = "s"

check = word.count(alphabet)
print("the presence of an alphabet 's' is :", check)

# out: the presence of an alphabet 's' is : 0
________________________________________



#B) How can you obtain 64 by using numbers 4 and 3

a = 4
b = 3
c = a**b #exponential a^b
print("the exponential of a,b is: ", c)

# out: the exponential of a,b is:  64
________________________________________
'''
# Module 3 - variables in Python 
# 1.	What will be the output of the following (can/cannot):
a.	Age1=5
b.	5age=55

2.	What will be the output of following (can/cannot):
a.	Age_1=100
b.	age@1=100

3.	How can you delete variables in Python ?
'''

Age1 = 5 #can 
print("age is:", Age1)
    
5age = 55 #can't #variable should not start with number

# out : SyntaxError: invalid syntax


#	What will be the output of following (can/cannot):
Age_1 = 100 #can
print("age is:", Age_1)

age@1 = 100 #can't #variable should not contain special character

out : SyntaxError: invalid syntax
    
3) How to delete variable 

 A = 45
 print("value of y is:", A)
 
 del A
print("value of x is", A)

    
#Module - 4 Conditional statements
Please write Python Programs for all the problems .
1.	 Take a variable ‘age’ which is of positive value and check the following:
a.	If age is less than 10, print “Children”.
b.	If age is more than 60 , print ‘senior citizens’
c.	 If it is in between 10 and 60, print ‘normal citizen’

2.	Find  the final train ticket price with the following conditions. 
a.	If male and sr.citizen, 70% of fare is applicable
b.	If female and sr.citizen, 50% of fare is applicable.
c.	If female and normal citizen, 70% of fare is applicable
d.	If male and normal citizen, 100% of fare is applicable
[Hint: First check for the gender, then calculate the fare based on age factor.. For both Male and Female ,consider them as sr.citizens if their age >=60]
3.	Check whether the given number is positive and divisible by 5 or not.  

#1) 	 Take a variable ‘age’ which is of positive value and check the following:
 

age = int(input("enter your age:"))

if age < 10:
    print("children")
if (age >10) and (age <60):
    print("normal citizen")
if age > 60:
    print("senior citizen")    
     
out: 	enter your age:27
normal citizen
    
#2 Find final ticket price    

fare = 250
female = 'f'
male = 'm'

gender = input("enter your gender: ")
age = int(input("enter your age:"))

if gender == female:
    
    if age >=60:
        
        price = fare*50/100
        print("the final train ticket price for female sr.citizen is:", price)

    else:
        price1 = fare*70/100
        print("the final train ticket price for female normal citizen: ", price1)

if gender == male:
    
    if age >=60:
        price2 = fare*70/100
        print("the final train ticket price for male sr. citizen: ", price2)
    
    else:
        price3 = fare*100/100
        print("the final train ticket price for male normal citizen: ", price3)
    
        out: enter your gender: male
enter your age:27
    
    #3 check given number is positive 
        
n = int(input("enter a number: "))

if n<0:
    print("the given number {} is negative:" .format(n))
elif n==0:
    print("the given number {} is zero: " .format(n))
else:
    print("the given number {} is positive: " .format(n))
    
    out: enter a number: 25
the given number 25 is positive:

#3 check the given number is divisible by 5 or not
num = int(input("enter a number: "))

if num % 5 == 0:
    print("the given number {} is  divisible by 5.." .format(num))
    
else:
    print("the given number {} is  not divisible by 5.." .format(num))    
    
    out: enter a number: 35
the given number 35 is  divisible by 5

out: enter a number: 52
the given number 52 is  not divisible by 5..
________________________________________


 # Module 5

1.	A) list1=[1,5.5,(10+20j),’data science’].. Print default functions and parameters exists in list1.
B) How do we create a sequence of numbers in Python.
C)  Read the input from keyboard and print a sequence of numbers up to that number

6.	Create 2 lists.. one list contains 10 numbers (list1=[0,1,2,3....9]) and other 
list contains words of those 10 numbers (list2=[‘zero’,’one’,’two’,.... ,’nine’]).
 Create a dictionary such that list2 are keys and list 1 are values..

6.	Consider a list1 [3,4,5,6,7,8]. Create a new list2 such that Add 10 to the even number and multiply with 5 if it is odd number in the list1..

6.	Write a simple user defined function that greets a person in such a way that :
6)	It should accept both name of person and message you want to deliver.
              ii) If no message is provided, it should greet a default message ‘How are you’
           Ex: Hello ---xxxx---, How are you  - default message.
            Ex: Hello ---xxxx---, --xx your message xx---




1.	A)
#Print Default Functions & parameters in List

list1 = [1,5.5,[10+20j], ‘data science’]
list2 = [25, ‘raj’, ‘True’]
dir(list1)   

out: 

[‘__add__’,
 ‘__class__’,
 ‘__contains__’,
 ‘__delattr__’,
 ‘__delitem__’,
 ‘__dir__’,
 ‘__doc__’,
 ‘__eq__’,
 ‘__format__’,
 ‘__ge__’,
 ‘__getattribute__’,
 ‘__getitem__’,
 ‘__gt__’,
 ‘__hash__’,
 ‘__iadd__’,
 ‘__imul__’,
 ‘__init__’,
 ‘__init_subclass__’,
 ‘__iter__’,
 ‘__le__’,
 ‘__len__’,
 ‘__lt__’,
 ‘__mul__’,
 ‘__ne__’,
 ‘__new__’,
 ‘__reduce__’,
 ‘__reduce_ex__’,
 ‘__repr__’,
 ‘__reversed__’,
 ‘__rmul__’,
 ‘__setattr__’,
 ‘__setitem__’,
 ‘__sizeof__’,
 ‘__str__’,
 ‘__subclasshook__’,
 ‘append’,
 ‘clear’,
 ‘copy’,
 ‘count’,
 ‘extend’,
 ‘index’,
 ‘insert’,
 ‘pop’,
 ‘remove’,
 ‘reverse’,
 ‘sort’]

list1.reverse()  #reversing the list          
print(list1) 
 
out: [‘data science’, [(10+20j)], 5.5, 1]
 	  
 	list1.append(45) #adding a value into the list
print(list1) 

out: ['d‘ta science',’[(10+20j)], 5.5, 1, 45]
________________________________________

  list1.copy()  #copy list
['d‘ta science',’[(10+20j)], 5.5, 1, 45]
________________________________________


list1.count(45) #count how many times value exist in the list
1
________________________________________

list1.extend(list2) #adding list into the list further
print(list1) 
out : ['d‘ta science',’[(10+20j)], 5.5, 1, 45, 25, 'r‘j',’True]
________________________________________


print(list1[3]) # accessing values of list using index
1
________________________________________

print(list1[1:4]) #accessing multiple values using index

out: [[(10+20j)], 5.5, 1]
________________________________________

list2[1] = 2+5j #updating the list value
print(list2)

[25, (2+5j), True]
________________________________________


list2.insert(2,2010) #inserting value into the list
print(list2) 
out: [25, (2+5j), 2010, True]
________________________________________


print(list2.pop(3)) #removing value using index
out: True
________________________________________


list2.index(‘True’) #getting the index of the value
out: 2
________________________________________

list1.clear() #deleting the list

len(list2)  # getting the length of the list    

out: 3
________________________________________


help(list2)    
dir(list2)    
    
        
#1)B) Create sequence of numbers in python
      
numbers = range(1, 10)
seq_num = []

for number in numbers:
        seq_num.append(number)
print(seq_num)        
    
out: [1, 2, 3, 4, 5, 6, 7, 8, 9]
________________________________________

#1)C) Read input from keyboard & print sequence of numbers

num = int(input("E“ter the number..")” 

seq_num1 = []


for number in range(1,num+1):
            seq_num1.append(number)

print(seq_num1)        

out: Enter the number..15
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

#2 create a dictionary such that list2 are keys and list1 are values

keys_list = ['z‘ro',’o’e',’t’o',’t’ree',’f’ur',’f’ve',’s’x',’s’ven',’e’ght',’n’ne']’values_list = [0,1,2,3,4,5,6,7,8,9]

zip_list = zip(keys_list,values_list) #get pairs of elements
dict1 = dict(zip_list)
print(dict1)

out: {'z‘ro':’0, 'o‘e':’1, 't‘o':’2, 't‘ree':’3, 'f‘ur':’4, 'f‘ve':’5, 's‘x':’6, 's‘ven':’7, 'e‘ght':’8, 'n‘ne':’9}
________________________________________


#3 create a list1 and add 10 to even number & multiply with 5 if it's’odd in the list1 

list1 = [3,4,5,6,7,8]
list2 = []
# output :list2 = [15,14,25,16,35,18]
for i In list1:
    if i%2==0:
        i = i+10
        list2.append(i)
    else:
        i = i*5
        list2.append(i)
print(list2)        

out: [15, 14, 25, 16, 35, 18]
________________________________________

#4 user defined function that greets a person 

name = input("E“ter your name: ")“msg = input("e“ter a message that you want: ")“    
def user():
    print(f'h’ {name} your message is {msg}')’
user()    

out: Enter your name: raju
enter a message that you want: hi
hi raju your message is hi

#) b) if no message provided it should greet a default message
name1 = input("E“ter your name: ")“
def user1():
    print(f”hello {name1} How are you”)

user1()    

out: 	Enter your name: jyoti
	hello jyoti How are you
________________________________________




 6 Module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\datasets\indian_cities.csv')
df.columns

#a)	Find out top 10 states in female-male sex ratio

top_sex_ratio = df.sort_values(by=['state_name','sex_ratio'], ascending=False).head(10)
top_sex_ratio



#b)	Find out top 10 cities in total number of graduates

top_grads = df.sort_values(by=['name_of_city','total_graduates'], ascending=False).head(10)
top_grads

 

# c)	Find out top 10 cities and their locations in respect of  total effective_literacy_rate.

top_cities = df.sort_values(by=['name_of_city','location','effective_literacy_rate_total'], ascending=False).head(10)
top_cities

 

#2) a)	Construct histogram on literates_total and comment about the inferences

plt.hist(df.literates_total) #data is not distributed normally
 
 
# b) scatter plot b/w male graduates and female graduates
 
plt.scatter(df.male_graduates, df.female_graduates)

 

#3) a) Box plot on total effective literacy rate

plt.boxplot(df.effective_literacy_rate_total) # Outliers are found at lower whisker
 

#b) find null values & delete them

df.isna().sum() # No null values found

name_of_city                      0
state_code                        0
state_name                        0
dist_code                         0
population_total                  0
population_male                   0
population_female                 0
0-6_population_total              0
0-6_population_male               0
0-6_population_female             0
literates_total                   0
literates_male                    0
literates_female                  0
sex_ratio                         0
child_sex_ratio                   0
effective_literacy_rate_total     0
effective_literacy_rate_male      0
effective_literacy_rate_female    0
location                          0
total_graduates                   0
male_graduates                    0
female_graduates                  0
dtype: int64

