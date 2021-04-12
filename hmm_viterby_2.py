# sumber:
# - https://github.com/saneshashank/HMM_POS_Tagging/blob/master/HMM_based_POS_tagging-applying%20Viterbi%20Algorithm.ipynb
# - https://github.com/famrashel/idn-tagged-corpus
# Importing libraries
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time

start = time.time()
print(time.strftime('%H:%M:%S', time.gmtime(start)))
# reading the Treebank tagged sentences
bukaf = open("korpus_idn.tsv")
dataf = bukaf.readlines()
nltk_data = []
kekata = []
for kata in dataf:
	dt = kata.replace('\n','').split('\t')
	if (len(dt) == 1):
		nltk_data.append(kekata)
		kekata = []
	else:
		kekata.append((dt[0],dt[1]))

# let's check some of the tagged data
print("contoh korpus")
print(nltk_data[0:5])
print("banyak kalimat :", len(nltk_data))

# split data into training and validation set in the ratio 80:20
train_set,test_set = train_test_split(nltk_data,train_size=0.8,test_size=0.2,random_state = None)#101)

# create list of train and test tagged words
train_tagged_words = [tup for sent in train_set for tup in sent]
test_tagged_words = [tup[0] for sent in test_set for tup in sent]

print('\nbanyaknya token dalam korpus')
print(len(train_tagged_words))
print(len(test_tagged_words))

# let's check how many unique tags are present in training data
tags = {tag for word,tag in train_tagged_words}

print("\nbanyak macam tag :", len(tags))

# let's check how many words are present in vocabulary
vocab = {word for word,tag in train_tagged_words}
print("banyak kata unik :", len(vocab))

'''====Build the Vanilla Viterbi based POS tagger===='''
# compute emission probability for a given word for a given tag
listag = {}
jlistag = {}
for tag in tags:
    listag[tag] = [pair[0] for pair in train_tagged_words if pair[1] == tag]
    jlistag[tag] = len(listag[tag])
def word_given_tag(word,tag,train_bag = train_tagged_words):
    taglist = listag[tag]
    tag_count = jlistag[tag]
    w_in_tag = [kata for kata in taglist if kata==word]    
    word_count_given_tag = len(w_in_tag)
    return (word_count_given_tag,tag_count)

# compute transition probabilities of a previous and next tag
def t2_given_t1(t2,t1,train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]    
    t1_tags = [tag for tag in tags if tag==t1]
    count_of_t1 = len(t1_tags)
    t2_given_t1 = [tags[index+1] for index in range(len(tags)-1) if tags[index] == t1 and tags[index+1] == t2]
    count_t2_given_t1 = len(t2_given_t1)    
    return(count_t2_given_t1,count_of_t1)

# creating t x t transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)
print('sedang menghitung..... (estimasi 1 tahun)')

tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)):
        t2_t1 = t2_given_t1(t2, t1)
        tags_matrix[i, j] = t2_t1[0]/t2_t1[1]
  #      print(tags_matrix[i, j])  memastikan program jalan

# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))

#matrix emisi
em_matrix = np.zeros((len(vocab), len(tags)), dtype='float32')
for i, kata in enumerate(list(vocab)):
    for j, tag in enumerate(list(tags)):
        kata_tag = word_given_tag(kata, tag)
        em_matrix[i, j] = kata_tag[0]/kata_tag[1]
em_df = pd.DataFrame(em_matrix, columns = list(tags), index = list(vocab))
        
print("\nmatrix transisi dalam korpus")
print(tags_df)
print("matrix emisi dalam korpus")
print(em_df)

# lets create a list containing tuples of POS tags and POS tag occurance probability, based on training data
tag_prob = []
total_tag = len([tag for word,tag in train_tagged_words])
for t in tags:
    each_tag = [tag for word,tag in train_tagged_words if tag==t]
    tag_prob.append((t,len(each_tag)/total_tag))

print('\n tag prob \n', tag_prob)

"""====Viterbi Algorithm===="""
# Vanilla Viterbi Algorithm
def Viterbi(words, train_bag = train_tagged_words):
    state = []
    state2 = []
    T = list(set([pair[1] for pair in train_bag]))
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] #Pe=[];
        Pt = []  #list probablitas transisi
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['Z', tag]
            else:
                transition_p = tags_df.loc[state[-1][0], tag]
                
            # compute emission and state probabilities
            try:
                emission_p = em_df.loc[word, tag] #akan eror jika tak ada word dalam corpus
            except:
                emission_p = 0  
            state_probability = emission_p * transition_p    
            p.append(state_probability)

            # find POS tag occurance probability
            tag_p = [pair[1] for pair in tag_prob if pair[0]==tag ]
            # calculate the transition prob weighted by tag occurance probability.
            transition_p = tag_p[0]*transition_p  
            Pt.append(transition_p)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]

         # if probability is zero (unknown word) then use transition probability
        if(pmax==0):
            pmax = max(Pt)
            state_max = T[Pt.index(pmax)]
            if (pmax==0):
                state_max = 'X'
        state.append([state_max, pmax])
        
        # viterby tag balik
    words.reverse()
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] #Pe=[];
        Pt = []  #list probablitas transisi
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc[tag, 'Z']
            else:
                transition_p = tags_df.loc[tag, state2[-1][0]]
                
            # compute emission and state probabilities
            try:
                emission_p = em_df.loc[word, tag] #akan eror jika takada word dalam corpus
            except:
                emission_p = 0  
            state_probability = emission_p * transition_p    
            p.append(state_probability)

            # find POS tag occurance probability
            tag_p = [pair[1] for pair in tag_prob if pair[0]==tag ]
            # calculate the transition prob weighted by tag occurance probability.
            transition_p = tag_p[0]*transition_p  
            Pt.append(transition_p)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]

         # if probability is zero (unknown word) then use transition probability
        if(pmax==0):
            pmax = max(Pt)
            state_max = T[Pt.index(pmax)]
            if (pmax==0):
                state_max = 'X'
        state2.append([state_max, pmax])
    state2.reverse()
    for k, status in enumerate(state):
        if status[0] != state2[k][0]:
            if status[1] < state2[k][1]:
                status[0] = state2[k][0]
    state1 = [status[0] for status in state]
    words.reverse()
    return list(zip(words, state1))

end = time.time()
difference = end-start
print('\n waktu training')
print(time.strftime('%H:%M:%S', time.gmtime(difference)))

'''===Testing Vanilla Viterbi Algorithm on sampled test data==='''
print('\n coba algoritma')
start = time.time()
# Let's test our Viterbi algorithm on a few sample sentences of test dataset
# list of untagged words
bnar = 0
sampel = 0
for sent in test_set:
    test_tagged_words = [tup[0] for tup in sent]
        # tagging the test sentences
    tagged_seq = Viterbi(test_tagged_words)
        # accuracy
    check = [i for i, j in zip(tagged_seq, sent) if i == j]
    bnar = bnar + len(check)
    sampel = sampel + len(sent)

end = time.time()
difference = end-start
print(end)
#print("Time taken in seconds: ", difference)
print(time.strftime('%H:%M:%S', time.gmtime(difference)))

accuracy = bnar/sampel
print('Vanilla Viterbi Algorithm Accuracy: ',accuracy*100)

# let's check the incorrectly tagged words
#print([j for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0] != j[1]])

"""===Evaluating tagging on sample 'Test_sentences.txt' file==="""
'''
#start = time.time()
f = open('tesKalimat.txt')
text = f.read()
test_sentences = text.splitlines()
f.close()
sample_test_sent = test_sentences[:-1]
print('\n'+"contoh kalimat:")
print(sample_test_sent)

# list of untagged words
kalim = []
for sent in sample_test_sent:
    sample_test_words = []
    for word in sent.split():
        sample_test_words.append(word.replace('.', '') )
    kalim.append(sample_test_words)
#sample_test_words = [word for sent in sample_test_sent for word in sent.split()]
print(kalim)

# tagging the test sentences
print('tunggu..... (100 tahun lagi)')
sample_tagged_seq = []
for sample_test_words in kalim:
   sample_tagged_seq.append(Viterbi(sample_test_words))
end = time.time()
difference = end-start

print('\n hasil:')
print(sample_tagged_seq)
print("Waktu yang diperlukan: ")
print(time.strftime('%H:%M:%S', time.gmtime(difference)))
'''
