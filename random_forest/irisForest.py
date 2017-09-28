from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
import os.path
from sklearn.metrics import accuracy_score
import numpy.random
%pylab inline

#Calculates the average of 50 decision trees --> random forest
#Sepal Length, Sepal Width, Petal Length, Sepal Width
iris = load_iris()
forestSum = 0 # sum of trees
treeCount = 50 
iterator =0 
sample = 100 # 2/3 of dataset
correctCount = 0
incorrectCount = 0
randIris = [] # random subset of Iris
chosenIndices =[]
randTarget = []
for i in range(treeCount): # find probability from each tree
    while iterator < sample: # choose 100 random arrays from iris data to analyze
        random_indice = randint(0, 149)
        if random_indice not in chosenIndices: #make sure same array isn't put in twice
            chosenIndices.append(random_indice)
            randIris.append(iris.data[random_indice])
            iterator += 1
            randTarget.append(iris.target[random_indice])
    iterator = 0         
    model = tree.DecisionTreeClassifier()
    model.fit(randIris, randTarget)
    newTarget = np.array(randTarget).reshape(-1, 1) # make 2D array
    while iterator < sample: # checks if prediction is correct
        prediction = int(round(randIris[iterator][argmin(randIris[iterator])]))
        target = newTarget[iterator]
        if (prediction == target):
            correctCount += 1
        else:
            incorrectCount += 1
        iterator+=1
    percentage = (correctCount/(correctCount + incorrectCount))
    forestSum += percentage
forestAvg = forestSum/treeCount
print(forestAvg)