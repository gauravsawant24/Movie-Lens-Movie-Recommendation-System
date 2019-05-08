import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pylab
%matplotlib inline
import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
import collections
conf = SparkConf().setMaster('local').setAppName('Test')
sc = SparkContext(conf=conf)
movielens = sc.textFile('ml-100k/u.data')
ratings = movielens.map(lambda x:x.split()[2])
most_movies = movielens.map(lambda x:x.split()[1])
##Most Ratings Histogram
stars = []
total_reviews = []
result = ratings.countByValue()
sorted_results = collections.OrderedDict(sorted(result.items()))
for key, value in sorted_results.items():
    stars.append('%s'%(key))
    total_reviews.append('%i'%(value))
d = {'Stars': stars, 'Reviews': total_reviews}
ratings.df = pd.DataFrame(data=d, dtype='int64')
sns.barplot(ratings.df['Stars'], ratings.df['Reviews'], hue = ratings.df['Stars'])
# Top 10 Movies having most ratings
Movie_id = []
reviews_t = []
result_2 = most_movies.countByValue()
sorted_results_2 = collections.OrderedDict(sorted(result_2.items()))
for key, value in sorted_results_2.items():
    Movie_id.append('%s'%(key))
    reviews_t.append('%i'%(value))
d_2 = {'Movie_id': Movie_id, 'Reviews': reviews_t}
movie_reviews = pd.DataFrame(data=d_2, dtype='int64')
movie_sub = movie_reviews.sort_values('Reviews', ascending=False)[0:10]
sns.factorplot(x = 'Movie_id', y = 'Reviews', hue = 'Movie_id', kind = 'bar', data = movie_sub,size = 6)
pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol = 10)