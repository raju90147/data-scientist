# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:26:00 2022

@author: LENOVO
"""
'''
Problem Statement: 
It is obvious that as part of data analysis we encounter a lot of text data which is a collection of strings which in turn is a sequence of characters. Access the text data and manipulate as per our requirements


1.	Create a string “Grow Gratitude”.
Code for the following tasks:
a)	How do you access the letter “G” of “Growth”?
b)	How do you find the length of the string?
c)	Count how many times “G” is in the string.

Sol: #create a string:-

g = "Grow Gratitude"

#a). access  g from the above string
g[0]


#b) length of string

len(g)

#c) count how many times 'G' is in the string

g.count('G')



2.	Create a string “Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else.”
Code for the following:
a)	Count the number of characters in the string.

Sol: string = "Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else."
    
#a) count the no.of characters in the string

len(string)


3.	Create a string "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"
Code for the following tasks:
a)	get one char of the word
b)	get the first three char
c)	get the last three char

Sol: s1 = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"

#a) get one char of word
s1[-1]

#b) get the 1st three char

s1[0:3]

#c) get last 3 char

s1[-3:]


4.	create a string "stay positive and optimistic". Now write a code to split on whitespace.
Write a code to find if:

The string starts with “H”
The string ends with “d”
The string ends with “c”

Sol: s2 = "stay positive and optimistic"

#a) string starts with "H"

s2.startswith("H ")

#B) String starts with "d"
s2.endswith("d")

#c) string starts with "c"
s2.endswith("c")


5.	Write a code to print " 🪐 " one hundred and eight times. (only in python)
Sol: print("🪐"*108)

6.	Write a code to print " o " one hundred and eight times. (only in R)


7.	Create a string “Grow Gratitude” and write a code to replace “Grow” with “Growth of”
Sol: g1 = "Grow Gratitude"
g1.replace('Grow', 'Growth of')

8.	A story was printed in a pdf, which isn’t making any sense. i.e.:
“.elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A”

Sol: story = "“.elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A”"
print(''.join(reversed(story)))



