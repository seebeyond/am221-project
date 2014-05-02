import json
import numpy as np
import matplotlib.pyplot as plt

review_file = '/Users/Michael/Documents/am221-data/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json'
f = open(review_file)
data = f.readlines()
f.close()

nr = len(data)
ns = 5
reviews = [json.loads(i) for i in data]

business_file = '/Users/Michael/Documents/am221-data/yelp_phoenix_academic_dataset/yelp_academic_dataset_business.json'
f = open(business_file)
data = f.readlines()
f.close()
businesses = [json.loads(i) for i in data]

checkin_file = '/Users/Michael/Documents/am221-data/yelp_phoenix_academic_dataset/yelp_academic_dataset_checkin.json'
f = open(checkin_file)
data = f.readlines()
f.close()
checkins = [json.loads(i) for i in data]

tip_file = '/Users/Michael/Documents/am221-data/yelp_phoenix_academic_dataset/yelp_academic_dataset_tip.json'
f = open(tip_file)
data = f.readlines()
f.close()
tips = [json.loads(i) for i in data]

user_file = '/Users/Michael/Documents/am221-data/yelp_phoenix_academic_dataset/yelp_academic_dataset_user.json'
f = open(user_file)
data = f.readlines()
f.close()
users = [json.loads(i) for i in data]




star_ct = np.zeros(ns)
star_list = np.zeros(nr)
funny = np.zeros(nr)
useful = np.zeros(nr)
cool = np.zeros(nr)

for i in range(nr):
  rev = reviews[i]
  txt = rev['text']
  stars = rev['stars']
  star_list[i] = stars
  funny[i] = rev['votes']['funny']
  useful[i] = rev['votes']['useful']
  cool[i] = rev['votes']['cool']
  star_ct[stars-1] += 1

star_pct = star_ct/float(nr)

print star_pct

# Histogram Plot
plt.figure()
# p0, = plt.plot([i+1 for i in range(5)], star_pct)
bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
plt.hist(star_list-0.5, bins, normed=True, histtype='bar', rwidth=0.8)
# p1, = plt.plot(time, x[:,1])
# plt.ylim([0, 1])
plt.xlim([0.5,5.5])
# plt.legend([p0, p1],["p","q"])
# plt.title('Game Theory Diff Eqn Model, [p0, q0] = [%f, %f]' % (xinit[0], xinit[1]))
plt.title('Histogram of Ratings')
plt.xlabel('Stars')
plt.ylabel('Frequency')
plt.show()

# Histogram Plot
plt.figure()
# bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
bins = range(71)
plt.hist(useful-0.5, bins, normed=True, histtype='bar', rwidth=0.8)
# plt.ylim([0, 1])
# plt.xlim([0.5,5.5])
# plt.legend([p0, p1],["p","q"])
# plt.title('Game Theory Diff Eqn Model, [p0, q0] = [%f, %f]' % (xinit[0], xinit[1]))
plt.title('Histogram of Useful')
plt.xlabel('Useful Votes')
plt.ylabel('Frequency')
plt.show()


# 
# # Line Plot
# plt.figure()
# p0, = plt.plot([i+1 for i in range(5)], star_pct)
# # p1, = plt.plot(time, x[:,1])
# plt.ylim([-0.1, 1.1])
# # plt.legend([p0, p1],["p","q"])
# # plt.title('Game Theory Diff Eqn Model, [p0, q0] = [%f, %f]' % (xinit[0], xinit[1]))
# plt.title('Distribution of Ratings')
# plt.xlabel('Stars')
# plt.ylabel('Frequency')
# plt.show()


# print 'Compiling Sochi Dict...'
# dict_sochi = {}
# for i in range(nr):
#   txt = tweets[i]['text'].lower()
#   words = re.split("[\W\s\!+\.+,+\?+\"]", txt)
#   words = filter(bool, words)
#   if i % 1000 == 0:
#     print(' Iteration %d' % (i))
#   for word in words:
#     if word in dict_sochi.keys():
#       dict_sochi[word] += 1 
#     else:
#       dict_sochi[word] = 1
# print ' Sorted %i tweets for %i words\n' % (ns, len(dict_sochi))


# reviews[0]['text']
# reviews[0]['stars']