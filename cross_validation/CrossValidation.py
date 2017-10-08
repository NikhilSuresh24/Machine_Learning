import pandas as pd
%pylab inline
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

data = pd.read_csv("~/Downloads/student/student-mat.csv",sep=';')

#create avg grade column
data['avg_grade'] = round((data['G1']+data['G2']+data['G3'])/3,0)

scores = []
accuracies = []
for folds in range(5):#repeats process for each test group
    start = int(79*folds)
    end = int(79*(i+folds))
    
    #puts together training set
    frames = [data.iloc[0:start][['failures','Medu', 'Fedu', 'traveltime','studytime','freetime', 'famrel','goout','Dalc','Walc','health','absences','avg_grade']],data.iloc[end:395][['failures','Medu', 'Fedu', 'traveltime','studytime','freetime', 'famrel','goout','Dalc','Walc','health','absences','avg_grade']]]
    modelData = pd.concat(frames)
    
    #testing set
    testData = data.iloc[start:end][['failures','Medu', 'Fedu', 'traveltime','studytime','freetime', 'famrel','goout','Dalc','Walc','health','absences','avg_grade']]
    
    #creating dataframes for variables
    y,X = dmatrices('failures~ Medu + Fedu + traveltime + studytime + famrel + freetime + goout + Dalc + Walc + health + absences + avg_grade', modelData, return_type = 'dataframe')
    
    #flatten y into a 1-D array
    y = np.ravel(y)
    
    #Fit model
    model = LogisticRegression()
    model = model.fit(X,y)
    
    #check model accuracy on training data
    scores.append(model.score(X,y))
    
    #check model acurracy on test data
    prediction = model.predict(testData)
    expected = testData.iloc[:]['failures']
    accuracies.append(metrics.accuracy_score(expected,prediction))
    
#create dataframe to display findings
df = pd.DataFrame({'0-78':pd.Series([scores[0],accuracies[0]], index=['Model Score', 'Accuracy Score']),
      '79-157':pd.Series([str(scores[1]),str(accuracies[1])],index=['Model Score', 'Accuracy Score']),
      '158-236':pd.Series([str(scores[2]),str(accuracies[2])], index=['Model Score', 'Accuracy Score']),
      '237-315':pd.Series([str(scores[3]), str(accuracies[3])], index=['Model Score', 'Accuracy Score']),
      '316-395':pd.Series([str(scores[4]), str(accuracies[4])],index=['Model Score', 'Accuracy Score']),
      'Average':pd.Series([str(mean(scores)),str(mean(accuracies))],index=['Model Score', 'Accuracy Score'])})

print(df)