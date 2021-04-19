#IMPORTS

import numpy as np
import pandas as pd

#DATA
column_name = ['user_id','item_id','rating','timestamp']

df = pd.read_csv('C:\\Users\\User\\Documents\\u.data',sep='\t',names=column_name)

print('\n',df.head(),'\n')

print('\n',df.info(),'\n')

movie_titles = pd.read_csv('C:\\Users\\User\\Documents\\Movie_Id_Titles')

print('\n',movie_titles.head(),'\n')

print('\n',movie_titles.info(),'\n')

movie_ratings = pd.merge(df,movie_titles,on='item_id')

print('\n',movie_ratings.head(),'\n')

#CHANGE WORKING DIRECTORY IF NECESSARY
import os
os.getcwd()
os.chdir('C:\\Users\\User\\Desktop\\school\\Python\\projects\\Recommender Systems')

#VISUALIZATION LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

#EDA
#Average ratings
print('\n',movie_ratings.groupby('title')['rating'].mean().head(),'\n')

print('\n',movie_ratings.groupby('title')['rating'].mean().sort_values(ascending=False).head(),'\n')

#We notice a lot of obscure movies, for the data to make sense, we need to filter the number of ratings

#Count of number of ratings per movie
print('\n',movie_ratings.groupby('title')['rating'].count().sort_values(ascending=False).head(),'\n')

ratings = pd.DataFrame(data=movie_ratings.groupby('title')['rating'].mean())

print('\n',ratings.head(),'\n')

ratings['num_of_ratings'] = movie_ratings.groupby('title')['rating'].count()

print('\n',ratings.head(),'\n')

#plot of number of ratings
ratings['num_of_ratings'].plot.hist(bins=70)
plt.savefig('Number of Ratings.jpg')
plt.show()

#Seaborn plot
i1 = sns.displot(data=ratings,x='num_of_ratings',bins=70)
i1.savefig('Seaborn Ratings Plot.jpg')
plt.show()

#100 ratings is a good threshold to filter recommendations by

#Distribution of ratings
ratings['rating'].plot.hist(bins=70)
plt.xlabel('Rating')
plt.savefig('Ratings.jpg')
plt.show()

ratings.head()
#Ratings vs Num of Ratings
i2 = sns.jointplot(data=ratings,x='rating',y='num_of_ratings',color='green')
i2.savefig('Ratings vs Number of Ratings.jpg')
plt.show()

movie_ratings.head()

movie_matrix = movie_ratings.pivot_table(values='rating',index='user_id',columns='title')

movie_matrix.head()

ratings.sort_values('num_of_ratings',ascending=False).head(10)

#Lets recommends movies similar to Star Wars and Scream
star_wars_user_rating = movie_matrix['Star Wars (1977)']
scream_user_rating = movie_matrix['Scream (1996)']

similar_to_starwars = movie_matrix.corrwith(star_wars_user_rating)
#IGNORE WARNING

similar_to_scream = movie_matrix.corrwith(scream_user_rating)
#IGNORE WARNING
#WE NOW HAVE Correlation to other movies
corr_starwars = pd.DataFrame(data=similar_to_starwars,columns=['Correlation'])
corr_starwars.sort_values('Correlation',ascending=False).head(10)

#The recommendation dont make sense because the num of ratings have not been accounted for
corr_starwars = corr_starwars.join(ratings['num_of_ratings'])
print('\n',corr_starwars[corr_starwars['num_of_ratings']>100].sort_values('Correlation',ascending=False).head(),'\n')

#Now we see our recommendations make sense

corr_scream = pd.DataFrame(data=similar_to_scream,columns=['Correlation'])
corr_scream = corr_scream.join(ratings['num_of_ratings'])
print('\n',corr_scream[corr_scream['num_of_ratings']>100].sort_values('Correlation',ascending=False).head(),'\n')

#Our recommendations make sense again based on just similar user ratings

#END




 
