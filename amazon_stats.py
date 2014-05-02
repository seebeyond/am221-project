import json
import numpy as np
import matplotlib.pyplot as plt
from jitter import *

# review_file = '/Users/Michael/Documents/am221-data/amazon/Kindle_Store.txt'
review_file = '/Users/Michael/Documents/am221-data/amazon/Software.txt'
f = open(review_file)
data = f.read()
f.close()

sp = data.split('\n\n')
sp = sp[:len(sp)-1]   # last one is blank
nr = len(sp)
ns = 5
rev_data = [sp[i].split('\n') for i in range(len(sp))]
help_num = np.zeros(nr)
help_den = np.zeros(nr)
score = np.zeros(nr)
score_ct = np.zeros(ns)
summary = ['' for i in range(nr)]
text = ['' for i in range(nr)]
text_len = np.zeros(nr)
text_wc = np.zeros(nr)

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

help_pct = help_num / help_den
score_pct = score_ct / nr
score_help = np.zeros(ns)
for i in range(ns):
  score_help[i] = np.mean(help_pct[np.logical_and(score == i+1.0, help_den >= 0.0)])

score_wc = np.zeros(ns)
for i in range(ns):
  score_wc[i] = np.mean(text_wc[score == i+1.0])

# Stats
print 'Number of reviews:', str(nr)
print 'Average number of chars:', np.mean(text_len)
print 'Average number of words:', np.mean(text_wc)
print 'Average score:', np.mean(score)
print 'Average helpfulness percentage:', np.mean(help_pct[help_den >= 0.0])
print 'Average helpful votes:', np.mean(help_num[help_den >= 0.0])
print 'Average total votes:', np.mean(help_den[help_den >= 0.0])
print 'Score frequencies:', score_pct
print 'Average score helpfulness:', score_help
print 'Average score word count:', score_wc
print 'Correlation between word count and helpfulness:', np.corrcoef(text_wc, help_pct)

# Histogram Plot
plt.figure()
# p0, = plt.plot([i+1 for i in range(5)], star_pct)
bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
plt.hist(score-0.5, bins, normed=True, histtype='bar', rwidth=0.8)
# p1, = plt.plot(time, x[:,1])
# plt.ylim([0, 1])
plt.xlim([0.5,5.5])
# plt.legend([p0, p1],["p","q"])
# plt.title('Game Theory Diff Eqn Model, [p0, q0] = [%f, %f]' % (xinit[0], xinit[1]))
plt.title('Histogram of Ratings')
plt.xlabel('Stars')
plt.ylabel('Frequency')
plt.show()

# Avg Wc Plot
plt.figure()
# p0, = plt.plot([i+1 for i in range(5)], star_pct)
# bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
plt.bar([i+0.5 for i in range(5)], score_wc)
# p1, = plt.plot(time, x[:,1])
# plt.ylim([0, 1])
plt.xlim([0.5,5.5])
# plt.legend([p0, p1],["p","q"])
# plt.title('Game Theory Diff Eqn Model, [p0, q0] = [%f, %f]' % (xinit[0], xinit[1]))
plt.title('Average Word Count per Score')
plt.xlabel('Score')
plt.ylabel('Average Word Count')
plt.show()


# Helpfulness plot
plt.figure()
# p0, = plt.plot([i+1 for i in range(5)], star_pct)
jitter(score, help_pct)
# p1, = plt.plot(time, x[:,1])
# plt.ylim([0, 1])
# plt.xlim([0.5,5.5])
# plt.legend([p0, p1],["p","q"])
# plt.title('Game Theory Diff Eqn Model, [p0, q0] = [%f, %f]' % (xinit[0], xinit[1]))
plt.title('Score vs Helpfulness')
plt.xlabel('Score')
plt.ylabel('Help Percentage')
plt.show()


# WC vs Helpfulness
plt.figure()
# p0, = plt.plot([i+1 for i in range(5)], star_pct)
jitter(text_wc, help_pct)
# p1, = plt.plot(time, x[:,1])
# plt.ylim([0, 1])
# plt.xlim([0.5,5.5])
# plt.legend([p0, p1],["p","q"])
# plt.title('Game Theory Diff Eqn Model, [p0, q0] = [%f, %f]' % (xinit[0], xinit[1]))
plt.title('Word Count vs Helpfulness')
plt.xlabel('Word Count')
plt.ylabel('Help Percentage')
plt.show()
