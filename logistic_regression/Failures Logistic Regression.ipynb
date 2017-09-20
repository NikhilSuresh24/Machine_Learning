
import pandas as pd
get_ipython().magic('pylab inline')
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# In this Logistic Regression, I am going to be predicting how many failures a student will get based off of other numerical factors

data = pd.read_csv("~/Downloads/student/student-mat.csv",sep=';')

#create avg grade column
data['avg_grade'] = round((data['G1']+data['G2']+data['G3'])/3,0)

data

data.groupby('age').mean()

data.groupby(['age','sex']).mean()

#data avgs
data.groupby('failures').mean()

#failures bar graph
data.failures.hist()
plt.title('Histogram of Failures')
plt.xlabel("Failures")
plt.ylabel("Frequency")

#how many students have each grade
data['avg_grade'].hist()
plt.title('Histogram of Average Grade')
plt.xlabel('Avg Grade')
plt.ylabel('Frequency')

#bar graph
pd.crosstab(data['avg_grade'],data.failures).plot(kind='bar')
plt.title('Grades and Failures')
plt.xlabel('Average Grade')
plt.ylabel('Number of Students')

#graph of avg grades and failures
grades = pd.crosstab(data['avg_grade'],data.failures)
grades.div(grades.sum(1), axis=0).plot(kind='bar', stacked = True)

#creating dataframes for variables
y,X = dmatrices('failures~ Medu + Fedu + traveltime + studytime + famrel + freetime + goout + Dalc + Walc + health + absences + avg_grade',data, return_type = 'dataframe')
print(X.columns)

#flatten y into a 1-D array
y = np.ravel(y)

model = LogisticRegression()
model = model.fit(X,y)

model.score(X,y)

#null error rate
1-y.mean()

#analyzing coefficients to improve regression
pd.DataFrame(X.columns, np.transpose(model.coef_[0]))

#creating groups based off of grades
df1 = pd.DataFrame(data, columns=['sex', "avg_grade"])
df2 = pd.DataFrame(data, columns=['sex', 'failures', 'Medu', 'Fedu', 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'])
bins = [0,4,9,14,20]
group_names = ['0-4','5-10','10-14','15-20']
grade_groups = pd.cut(df1['avg_grade'],bins, labels = group_names)
df1['grade_groups']= pd.cut(df1['avg_grade'],bins, labels = group_names)
df2.head(10)

df2['grade_groups'] = df1['grade_groups']
df2['avg_grade'] = df1['avg_grade']
df2

y,Z = dmatrices('failures~ Medu + Fedu + traveltime + studytime + famrel + freetime + goout + Dalc + Walc + health + absences + grade_groups',data, return_type = 'dataframe')
print(Z.columns)

y = np.ravel(y)

Z.head(100)

model1 = LogisticRegression()
model1 = model1.fit(Z,y)

model1.score(Z,y)
y.mean()
# The model had a 78.48% chance of success, versus a 66.6% null error rate, so it is 12% better than just guessing the same number of failures every time. Additionally, grouping people by grades did not improve the model.
