#########################################################################################
# Header and Imports
#########################################################################################

import json
import numpy as np
import random
import re
# from stemming.porter2 import stem   # stemming library
import time
import csv
from collections import Counter
import subprocess

from constants import *


#########################################################################################
# Define constants and parameters
#########################################################################################
'''
# number of rating categories
ns = 5

# number of reviews in learning set
nl = 1000

# minimum number of occurrences in corpus to be considered for feature
min_ct = 20

# number of most common words to remove
ncommon = 100

# number of top significant words for each category to use as features
ntop = 100

# penalty factor for SVM
C = 10

# file containing review data
review_file = './raw_data/Software.txt'
# review_file = '/Users/Michael/Documents/am221-data/amazon/Software.txt'
# review_file = '/Users/Michael/Documents/am221-data/amazon/Kindle_Store.txt'   

# AMPL file directory name
ampl_folder = './ampl_svm/'

# AMPL runfile
ampl_run = ampl_folder + 'svm.run'

# SVM directory
svm_folder = './svm_data/'
'''

#########################################################################################
# Read in Amazon data and get it in usable form
#########################################################################################

print 'Reading in data...'
start = time.time()

# read in data
f = open(review_file)
data = f.read()
f.close()

# separate reviews
sp = data.split('\n\n')
sp = sp[:len(sp)-1]   # last one is blank

# number of reviews
nr = len(sp)

# get list of different review parts
rev_data = [sp[i].split('\n') for i in range(len(sp))]

# initialize storage
help_num = np.zeros(nr)
help_den = np.zeros(nr)
score = np.zeros(nr)
score_ct = np.zeros(ns)
summary = ['' for i in range(nr)]
text = ['' for i in range(nr)]
text_len = np.zeros(nr)
text_wc = np.zeros(nr)

# store review data
for i in range(nr):
    help = rev_data[i][5][20:].split('/')
    help_num[i] = float(help[0])
    if int(help[1]) == 0:
        help_den[i] = -1.0
    else:
        help_den[i] = float(help[1])
    score[i] = float(rev_data[i][6][14:])
    score_ct[int(float(rev_data[i][6][14:]))-1] += 1
    summary[i] = rev_data[i][8][16:] 
    text[i] = rev_data[i][9][13:]
    text_len[i] = len(rev_data[i][9][13:])
    text_wc[i] = len(rev_data[i][9][13:].split())

end = time.time()
print ' Time elapsed: %.4f sec' % (end - start)


#########################################################################################
# Construct dictionary from random sample
#########################################################################################

print 'Constructing learning dictionaries...'
start = time.time()

### Comment this section out after first time running for increased speed

# randomly select learning set
random.seed(0)
learn_idx = random.sample(range(nr), nl)

# initialize dictionaries for each score category
learn_dicts = [{} for i in range(ns)]
dict_all = {}

# populate dictionaries from learning set
for i in range(nl):
    idx = learn_idx[i]
    scr = int(score[idx]) - 1
    # get rid of escaped characters and apostrophes
    txt = text[idx].lower().replace("&quot;","'").replace("'","")
    # reg-ex to split on
    words = re.split("[\W\s\!+\.+,+\?+\"]", txt)
    words = filter(bool, words)
#     if i % 1000 == 0:
#       print(' Iteration %d' % (i))
    for word in words:
#       word = stem(word)     # extract root of word
      # add to category dictionary
        if word in learn_dicts[scr].keys():
            learn_dicts[scr][word] += 1 
        else:
            learn_dicts[scr][word] = 1
        # add to global dictionary
        if word in dict_all.keys():
            dict_all[word] += 1 
        else:
            dict_all[word] = 1

# save dictionaries so for faster processing
for i in range(ns):
    f = open("dicts/learn_dict" + str(i + 1) + ".csv", "w")
    w = csv.writer(f)
    for key, val in learn_dicts[i].items():
        w.writerow([key, val])
    f.close()
# global dictionary
f = open("dicts/dict_all.csv", "w")
w = csv.writer(f)
for key, val in dict_all.items():
    w.writerow([key, val])
f.close()

### End commented out section

# read the dictionaries in
learn_dicts = [{} for i in range(ns)]
dict_all = {}
for i in range(ns):
    f = open("dicts/learn_dict" + str(i + 1) + ".csv", "r")
    for key, val in csv.reader(f):
        learn_dicts[i][key] = int(val)
    f.close()
# global dictionary
f = open("dicts/dict_all.csv", "r")
for key, val in csv.reader(f):
    dict_all[key] = int(val)
f.close()

# read in most common words
# see http://norvig.com/ngrams/
f = open("data/count_1w.txt")
dict_common = {}
tmp = 0
for l in f:
    if tmp >= ncommon:
        break
    entry = l.split()
    dict_common[entry[0]] = 1
    tmp += 1
f.close()

end = time.time()
print ' Time elapsed: %.4f sec' % (end - start)


#########################################################################################
# Select features to use
#########################################################################################

print 'Selecting features...'
start = time.time()

# Calculate z-scores for each feature
# see http://www.asso-aria.org/coria/2012/273.pdf for details
score_dicts = [{} for i in range(ns)]
for i in range(ns):
    Cm = dict(Counter(dict_all) - Counter(learn_dicts[i]))
    aplusc = sum(learn_dicts[i].values())
    bplusd = sum(Cm.values())
    n = aplusc + bplusd
    for key, val in learn_dicts[i].items():
        f = key
        # throw out if total number of occurrences is less than min_ct
        if dict_all[f] <= min_ct or f in dict_common.keys():
            continue
        a = val     # a = learn_dicts[i][f]
        if f in Cm.keys():
            b = Cm[f]
        else:
            b = 0
        Pf = (a+b)/float(n)
        Zf = (a - aplusc * Pf)/np.sqrt(aplusc * Pf * (1-Pf))
        score_dicts[i][f] = Zf

# get words with highest z-score in each category
sd = [[] for i in range(ns)]
for i in range(ns):
    sd[i] = sorted(score_dicts[i].items(), key=lambda x: -x[1])
#     print ' ' + str(i+1) + ' stars:'
#     ct = 0
#     for tuple in sd:
#         if ct >= 20:
#             break
#         ct += 1
#       print '  ', tuple[0],':',tuple[1]

# build the svm dictionary
svm_dict = {}
idx = 0
for i in range(ns):
    end_idx = ntop if ntop < len(sd[i]) else len(sd[i])   # ensure no index error
    for k, v in sd[i][0:end_idx]:
        if k not in svm_dict.keys():
            svm_dict[k] = idx
            idx += 1
# number of features
nf = idx

end = time.time()
print ' Time elapsed: %.4f sec' % (end - start)


#########################################################################################
# Build the feature vectors for one-vs-all approach
#########################################################################################

print 'Building feature vectors...'
start = time.time()

# lists for various categories
X_list = [np.zeros((nl,nf)) for i in range(ns)]
Y_list = [np.zeros((nl,1)) for i in range(ns)]

# loop over categories
for cat in range(ns):
    # initialize objects
    X = np.zeros((nl,nf))
    Y = np.zeros((nl,1))
    row = 0

    # loop over learning reviews
    for i in range(nl):
        idx = learn_idx[i]
        scr = int(score[idx]) - 1
        # get rid of escaped characters and apostrophes
        txt = text[idx].lower().replace("&quot;","'").replace("'","")
        # reg-ex to split on
        words = re.split("[\W\s\!+\.+,+\?+\"]", txt)
        words = filter(bool, words)
        for word in words:
#             word = stem(word)     # extract root of word
            if word in svm_dict.keys():
                X[row, svm_dict[word]] += 1
        Y[row] = 1 if scr == cat else -1
        row += 1
    
    # save objects
    X_list[cat] = X
    Y_list[cat] = Y

end = time.time()
print ' Time elapsed: %.4f sec' % (end - start)


#########################################################################################
# Write AMPL data file and solve SVM optimization problem using AMPL
#########################################################################################

print 'Writing and solving AMPL data files...'

# loop over categories
for cat in range(ns):
    print ' Building %d star-vs-all SVM...' % (cat+1)
    start = time.time()
    
    # get data objects
    X = X_list[cat]
    Y = Y_list[cat]
    
    # write AMPL runfile
    f = open(ampl_run, 'w')
    f.write('model ' + ampl_folder + 'svm.mod;\n')
    f.write('data ' + ampl_folder + 'svm_data' + str(cat+1) + '.dat;\n')
    f.write("option solver 'cplex';\n")
    f.write("option solver_msg 0;\n")
    f.write("solve;\n");
    f.write("option display_1col 100000000;\n")
    f.write("display w > " + ampl_folder + "svm-w.txt;\n")
    f.write("display b > " + ampl_folder + "svm-b.txt;\n")
    f.close()
    
    # open data file
    filename = ampl_folder + 'svm_data' + str(cat+1) + '.dat'
    f = open(filename,'w')
    
    # Y parameter string
    sy = ''
    for i in range(nl):
        sy = sy + ' ' + str(i+1) + ' ' + np.array_str(Y[i,0])
    
    # X parameter string
    sx = 'param X :'
    # initial line
    for i in range(nf):
        sx = sx + ' ' + str(i + 1)
    # end of first line
    sx = sx + ' :=\n'
    # create matrix body
    for i in range(nl):
        sx = sx + '          ' + str(i + 1)
        for j in range(nf):
            sx = sx + ' ' + np.array_str(X[i,j])
        sx = sx + '\n'
    # take off last break and add in semicolon
    sx = sx[:-1] + ';\n'
    
    # write lines
    f.write('data;\n')
    f.write('param n := ' + str(nl) + ';\n')
    f.write('param d := ' + str(nf) + ';\n')
    f.write('param C := ' + str(C) + ';\n')
    f.write('param Y := ' + sy + ';\n')
    f.write(sx)
    
    # close data file
    f.close()
    
    # run ampl
    devnull = open('/dev/null', 'w')
    _ = subprocess.call(["ampl", ampl_run], stdout=devnull, stderr=devnull)
    devnull.close()
    
    # read in svm data from AMPL output
    w = np.zeros(nf)
    b = np.zeros(1)
    # get w values
    f = open(ampl_folder + "svm-w.txt")
    _ = f.readline()    # dump first line
    idx = 0
    for l in f:
        tmp = l.replace('\n','').split(' ')
        tmp = filter(bool, tmp)
        if idx < nf:
            w[idx] = float(tmp[1])
        idx += 1
    f.close()
    # get b value
    f = open(ampl_folder + "svm-b.txt")
    tmp = f.readline()
    tmp = tmp.replace('\n','').replace('=', '').split(' ')
    tmp = filter(bool, tmp)
    b[0] = float(tmp[1])
    f.close()
    
    # save svm
    out_file_w = svm_folder + str(cat+1) + '-svm-w.txt'
    out_file_b = svm_folder + str(cat+1) + '-svm-b.txt'
    np.savetxt(out_file_w, w)
    np.savetxt(out_file_b, b)
    
    end = time.time()
    print ' Time elapsed: %.4f sec' % (end - start)
# end loop

# save svm dictionary
out_file_idx = svm_folder + 'svm-dict.csv'
f = open(out_file_idx, "w")
w = csv.writer(f)
for key, val in svm_dict.items():
    w.writerow([key, val])
f.close()


#########################################################################################
# Make predictions
#########################################################################################

print 'Making a prediction for a random review...'
start = time.time()

# get random index
pidx = random.sample(range(nr),1)[0]

# initialize lists for SVMs
w_list = [np.zeros(nf) for i in range(ns)]
b_list = [np.zeros(1) for i in range(ns)]
conf_scores = np.zeros(ns)

# import SVM dict
svm_dict = {}
in_file_idx = svm_folder + 'svm-dict.csv'
f = open(in_file_idx)
for key, val in csv.reader(f):
    svm_dict[key] = int(val)
f.close()

# construct feature vector for the test
x = np.zeros(nf)
scr = int(score[pidx]) - 1
# get rid of escaped characters and apostrophes
txt = text[pidx].lower().replace("&quot;","'").replace("'","")
# reg-ex to split on
words = re.split("[\W\s\!+\.+,+\?+\"]", txt)
words = filter(bool, words)
for word in words:
    if word in svm_dict.keys():
#         print word
        x[svm_dict[word]] = 1

for i in range(ns):
    # SVM files
    in_file_w = svm_folder + str(i+1) + '-svm-w.txt'
    in_file_b = svm_folder + str(i+1) + '-svm-b.txt'
    # import SVM and save
    w = np.loadtxt(in_file_w)
    b = np.loadtxt(in_file_b)
    w_list[i] = w
    b_list[i] = b
    
    # get confidence score for i-vs-all SVM
    c = np.dot(w,x) - b
    conf_scores[i] = c

# Print prediction and data
predict = np.argmax(conf_scores)
print ' Predicted score is %d' % (predict + 1)
print ' Actual score is %d' % (scr + 1)
print ' Confidence scores:', conf_scores
print ' Review text:'
print ' ' + text[pidx]
print

end = time.time()
print ' Time elapsed: %.4f sec' % (end - start)
