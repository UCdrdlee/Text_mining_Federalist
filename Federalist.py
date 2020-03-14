import urllib.request
import numpy as np
import pandas as pd
import re
pd.set_option('display.max_rows',None)
url='http://www.stat.uchicago.edu/~meiwang/courses/pg18.txt'
urllib.request.urlretrieve(url, '/Users/dongillee/Downloads/pg18.txt')

### PART2 ###

fedvol=[]
lines=[]
pattern = re.compile(r'FEDERALIST\.?\sNo.\w*')  # Pattern to match.
with open('pg18.txt','r') as pg18:
    contents = pg18.read()
    fed = pattern.findall(contents) # Store all the matching phrases
    # Store start and end indices for all the matching phrases
    indices =[(m.start(0), m.end(0)) for m in re.finditer(pattern, contents)]
    indices=np.array(indices,dtype=np.dtype('int')) # Conver a list of tuples to np array
    for i in range(len(indices)):
        # Iterate through all the matching phrases to print out the whole titles as well as store them for later use
        if i<9:
            print(contents[indices[i][0]:indices[i][0]+len(fed[i])+2])
            fedvol.append(contents[indices[i][0]:indices[i][0]+len(fed[i])+2])
        else:
            print(contents[indices[i][0]:indices[i][0]+len(fed[i])+3])
            fedvol.append(contents[indices[i][0]:indices[i][0]+len(fed[i])+3])

# This function takes a phrase and print and return a line number where the phrase is found
def get_line_number(phrase, file_name):
    with open(file_name) as f:
        for i, line in enumerate(f, 1):
            if phrase in line:
                #print(i)
                return i

for i in range(len(fedvol)):
    #print("The line number for ", fedvol[i])
    line=get_line_number(fedvol[i],'pg18.txt')
    lines.append(line)

linenum = np.array(lines)
linenum=np.delete(linenum,70,0) # Drop duplicate article No. 70
line_num=pd.DataFrame(linenum,index=range(1,86),columns=['line number'])
line_num

### PART4 ###

# Define patterns
madison = re.compile(r'MADISON')
hamilton = re.compile(r'HAMILTON')
jay=re.compile(r'JAY')
Ham_mad = re.compile('HAMILTON\sAND\sMADISON|HAMILTON\sOR\sMADISON')
indices=np.delete(indices,70,0) # remove duplicate article 70.

zeros = np.zeros((85,))
authorships = pd.DataFrame(zeros,index=range(1,86),columns=['author']) # Initialize dataframe
# This function will take the pattern as input and find if there is a match in a short segment of the ebook.
# If match is found, update the dataframe with the author's name
def which_article(pattern):
    for i in range(len(indices)):
        search=contents[indices[i][1]:indices[i][1]+300] # get a short segment of ebook based on indices
        author = pattern.findall(search) # find match
        if len(author)!=0:
            authorships.iloc[i,0]=author[0] # if match found, update the dataframe
# Feed all the patterns into the above function
which_article(madison)
which_article(jay)
which_article(hamilton)
which_article(Ham_mad)

# Visualize the authorships for the articles
pd.set_option('display.max_rows',None)
print(authorships)

#### Question 4: Get the counts
# Store article indices single-authored by Hamilton and Madison
H=authorships.loc[authorships['author'] == 'HAMILTON'].index.tolist()
M=authorships.loc[authorships['author'] == 'MADISON'].index.tolist()
print("The number of articles single-authored by Hamilton is: ",len(H))
print("The number of articles single-authored by Hamilton is: ",len(M))

### PART5 ###

# We first concatenate the authorships and line_num dataframe to get complete information
frames = [authorships,line_num]
auth_line=pd.concat(frames,axis=1)

# This function takes a list of article numbers, count the number of words in each article and return avg of them.
def wordsinarticle(listofarticles):
    number_words=[]
    for article in listofarticles:
        start = auth_line.loc[article,'line number'] # get line number that the article begins
        end = auth_line.loc[article+1,'line number'] # get line number that the next article begins
        # get the article segment defined to be the segment between 'article start' and 'next_start'
        segment = open("pg18.txt", "r").readlines()[start:end] # get segment of the txt file corresponding to the article
        num_words = [len(sentence.split()) for sentence in segment] # chop the segment into words
        sum_words = sum(num_words)
        number_words.append(sum_words)
    return sum(number_words)/len(listofarticles)

# Hamilton's articles' average number of words:
avg_Hamilton=wordsinarticle(H[0:50])
print("Average number of words in articles single-authored by Hamilton is approximately: ", avg_Hamilton)
# Madison's articles' average number of words:
avg_Madison=wordsinarticle(M)
print("Average number of words in articles single-authored by Madison is approximately: ", avg_Madison)

### PART6 ###

# We first begin by creating a dataframe and populating the slots with known information
# Note that here I'm excluding the FEDALIST No.85 by Hamilton because getting the last line of this article
# requires additional coding. Given that we already have 50 articles, we assume adding another one wouldn't
# affect the result of the subsequent regression and prediction
columns = ['length','by','enough','from','this','that','to','upon','while','whilst','author']
zeros = np.zeros((65,11))
index=H[0:50]+M
table = pd.DataFrame(zeros, index=index, columns=columns) # Initialize the dataframe
table.iloc[0:50,10]=0 # First 50 rows are for Hamilton
table.iloc[51:,10]=1 # Next 15 rows are for Madison

import string
# Return a dictionary with keys being words and values being the number of the corresponding words
def word_counter(words):
    words_said={}
    for word in words:
        if word not in words_said:
            words_said[word]=0
        if word in words_said:
            words_said[word]+=1
    return words_said

def count_words(article):
    start = auth_line.loc[article,'line number'] # get line number that the article begins
    end = auth_line.loc[article+1,'line number'] # get line number that the next article begins
    # get the article segment defined to be the segment between 'article start' and 'next_start'
    segment = open("pg18.txt", encoding='utf-8', errors='ignore').readlines()[start:end]
    words = [sentence.split() for sentence in segment]
    words = [item for sublist in words for item in sublist] # flatten list of lists into a single list
    words = [x.lower() for x in words] # change all words to lowercase
    words = [s.translate(str.maketrans('','',string.punctuation)) for s in words] # remove punctuations
    words=word_counter(words) # feed the list of words into word_counter function
    return words

# This function returns the number of matched keyword. If not found any, return 0
def get_keyword(word_count,keyword):
    if keyword in word_count.keys():
        return cw[keyword]
    else:
        return 0

# This functions takes an article and return number of words used in that article
def length(article):
    #for article in listofarticles:
    start = auth_line.loc[article,'line number'] # get line number that the article begins
    end = auth_line.loc[article+1,'line number'] # get line number that the next article begins
    # get the article segment defined to be the segment between 'article start' and 'next_start'
    segment = open("pg18.txt", "r").readlines()[start:end]
    num_words = [len(sentence.split()) for sentence in segment]
    sum_words = sum(num_words)
    return sum_words

# Fill in the datafram 'Table' with the number of 9 keywords used by Hamilton
for i in range(len(H[0:50])):
    cw=count_words(H[i])
    table.iloc[i,0]=length(H[i])
    for j in range(1,10):
        table.iloc[i,j]=get_keyword(cw,columns[j])

# Fill in the datafram 'Table' with the number of 9 keywords used by Madison
for i in range(len(M)):
    cw=count_words(M[i])
    table.iloc[i+50,0]=length(M[i])
    for j in range(1,10):
        table.iloc[i+50,j]=get_keyword(cw,columns[j])

from sklearn.linear_model import LogisticRegression
parameters = np.array(table.iloc[:,0:9]) # set values for 10 key features as parameters
response = table.iloc[:,10].tolist() # set label '1' or '0' as responses
# logistic fit to the training data
clf = LogisticRegression(random_state=0).fit(parameters, response)
# Print out the coefficients for the parameter (parameter estimates)
print("Coefficients of fit logistic curve is: ", clf.coef_)

### PART7 ###

# get article number authored by 'Hamilton or Madison'
HM=authorships.loc[authorships['author'] == 'HAMILTON OR MADISON'].index.tolist()
zeros = np.zeros((len(HM),10))
HorM = pd.DataFrame(zeros, index=HM, columns=columns[0:10]) # Initialize the dataframe

# Fill in the datafram 'HM' with the number of 9 keywords used by Hamilton or Madison
for i in range(len(HM)):
    cw=count_words(HM[i])
    HorM.iloc[i,0]=length(HM[i])
    for j in range(1,10):
        HorM.iloc[i,j]=get_keyword(cw,columns[j])

parameters2 = np.array(HorM.iloc[:,0:9]) # set values for 10 key features as parameters
prob=LogisticRegression.predict_proba(clf,parameters2)
print("Probability for Hamilton(0) vs Madison(1) is as follows:")
print(pd.DataFrame(prob,index=HM))
print("pridiction is: ", LogisticRegression.predict(clf,parameters2))
print("Madison wrote all the 'HAMILTON or MADISON' articles based on the result")
