import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as plt
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#Create your df here:
cupid = pd.read_csv("profiles.csv")

#print(cupid.keys())
#print(cupid.job.value_counts())


#Mapping for new columns
kids = {
	'doesn&rsquo;t have kids': int(0),
	'doesn&rsquo;t have kids, but might want them': int(0),
	'doesn&rsquo;t have kids, and doesn&rsquo;t want any': int(0),
	'doesn&rsquo;t have kids, but wants them': int(0),
	'doesn&rsquo;t want kids': int(0),
	'wants kids': int(0),
	'might want kids': int(0),
	'has kids'   : int(1),
	'has a kid': int(1),
	'has kids, but doesn&rsquo;t want more': int(1),
	'has a kid, but doesn&rsquo;t want more': int(1),
	'has a kid, and might want more': int(1),
	'has kids, and might want more': int(1),
	'has a kid, and wants more': int(1),
	'has kids, and wants more': int(1),
}

statuses = {
	'single': 1,
	'seeing someone': 2,
	'available': 1,
	'married': 2
}

gender = {
	'f': 1,
	'm': 2
}

orietation = {
	'straight': 1,
	'gay': 2,
	'bisexual': 3
}

drink = {
	'not at all': 1,
	'rarely': 2,
	'socially': 3,
	'very often': 4,
	'desperately': 5
}

drugged = {
	'never':1,
	'somtimes':2,
	'often':3
}

body_types = {
	'used up':1,
	'skinny':2,
	'thin' :3,
	'fit':4,
	'athletic':5,
	'jacked':6,
	'average':7,
	'a little extra':8,
	'full figured':9,
	'curvy':10,
	'overweight':11,
	'rather not say':12
}

jobs = {
	'other': 1,
	'student': 2,
	'science / tech / engineering': 3,
	'computer / hardware / software': 4,
	'artistic / musical / writer': 5,
	'sales / marketing / biz dev': 6,
	'medicine / health': 7,
	'education / academia': 8,
	'executive / management': 9,
	'banking / financial / real estate': 10,
	'entertainment / media': 11,
	'law / legal services': 12,
	'hospitality / travel': 13,
	'construction / craftsmanship': 14,
	'clerical / administrative': 15,
	'political / government': 16,
	'rather not say': 17,
	'transportation': 18,
	'unemployed': 19,
	'retired': 20,
	'military': 21,
}
#Creating new columns
cupid['kids'] = cupid.offspring.map(kids)
cupid['statuses'] = cupid.status.map(statuses)
cupid['gender'] = cupid.sex.map(gender)
cupid['orietation'] = cupid.orientation.map(orietation)
cupid['drink'] = cupid.drinks.map(drink)
cupid['body_types'] = cupid.body_type.map(body_types)
cupid['drugged'] = cupid.drugs.map(drugged)
cupid['jobs'] = cupid.job.map(jobs)

#Grouping the Data and dropping NaNs
data = cupid[[
	'age',
	'kids', 
	'statuses', 
	'gender', 
	'orietation', 
	'drink', 
	'body_types',
	'drugged',
	'jobs'
]].dropna()

#Normalizing the Data
e = data.values
scaler = preprocessing.MinMaxScaler()
np_scaled = scaler.fit_transform(e)
data = pd.DataFrame(np_scaled, columns=data.columns)
print(data)
#Training and Testing Data
x = data[['age', 'body_types', 'statuses', 'orietation', 'jobs']]
y = data[['kids']]

#Splitting Train/Test Data 
x_train, x_test, y_train, y_test = train_test_split(x, y,
							 train_size = 0.8, test_size = 0.2, 
								random_state = 100)

lr = LinearRegression()

#Fitting the Data and predicting Y for LinearRegression
lr_y_predicted = lr.fit(x_train, y_train).predict(x_test)
print("The score of your Linear Regression model is %f" % (lr.score(x_test, y_test)))
print("Here are the weightings of your Linear Regression model: \n%a" % (lr.coef_))

#Fitting the data for NaiveBayes
gnb = GaussianNB()

gnb_y_predicted = gnb.fit(x_train, y_train.values.ravel()).predict(x_test)
print("The score of your Naive Bayes model is %f" % (gnb.score(x_test, y_test)))

##Plotting people who have kids vs people who don't
##Unsurprisingly most people looking for a relationship do not have children
##Can I predict whether someone on OKCupid has kids or not?
cupid.kids.value_counts().plot(kind='bar', 
			title='Users with Offspring')
plt.xticks(ticks=[0,1], labels=[
			"Don't Have Offspring", 
			"Have Offspring"
			], rotation='horizontal')
plt.yticks(ticks=[4919, 19466])
plt.show()

#Labels for Body Type Pie Chart
labels = ([
				'Used up',
				'Skinny',
				'Thin' ,
				'Fit',
				'Athletic',
				'Jacked',
				'Average',
				'A Little Extra',
				'Full Figured',
				'Curvy',
				'Overweight',
				'Rather Not Say'])
explode=[0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4]
#Body Type Pie Chart
cupid.body_types.value_counts().plot(kind='pie', 
				title="Body Types", labels=labels, wedgeprops={'linewidth':100},
				explode=explode, rotatelabels=False)
plt.ylabel('')

plt.show()
