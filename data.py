# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib
print('Libraries loaded.')

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print('Dataset loaded.')

#instances (rows) and attributes (columns)
print('Dataset shape: ',(dataset.shape))


print('*******************************************************************************')
#first 20 rows of the dataset
print('First 20 rows of the dataset:')
print(dataset.head(20))



print('*******************************************************************************')
#summary of each attribute (column)
print(dataset.describe())



print('*******************************************************************************')
#class distribution
print(dataset.groupby('class').size())



print('*******************************************************************************')

#box and whisker plots for each variable
#subplots - each variable will be in a separate plot
#layout = (2,2) - 2 rows and 2 columns
#sharex = False - each plot will have its own x axis (sharey = for y axis)
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()

#gistograms of each variable
dataset.hist()
plt.show()

#interactions between variables- scatter plot
scatter_matrix(dataset)
plt.show()


print('*******************************************************************************')

#validation dataset - split into 80% to train and 20% to test

array = dataset.values
x = array[:,0:4]
y =array[:,4]
x_train, x_validation   , y_train, y_validation = train_test_split(x,y, test_size = 0.2, random_state = 1)


print('Train set: ', x_train.shape, y_train.shape)
print('Validation set: ', x_validation.shape, y_validation.shape)



print('*******************************************************************************')

#Test algorithms- simple linear and non linear algorithms
models = []
models.append(('LR', LogisticRegression(max_iter = 200))) #probability of data being in a class
models.append(('LDA', LinearDiscriminantAnalysis())) #avg of species projected to a line and then classified
models.append(('KNN', KNeighborsClassifier()))       #based on similarity to neighbours
models.append(('CART', DecisionTreeClassifier()))    #based on a tree x yes or no questions 
models.append(('NB', GaussianNB()))                  #based on measurements, each value being standalone
models.append(('SVM', SVC(gamma = 'auto')))          #widest safe margin between classes

#evalutating each model in turn
results = []
names = []

for name, model in models:
    #train on 9 parts and test on 1, randomlu, and scrambe data
    kfold = StratifiedKFold(n_splits= 10, random_state = 1, shuffle = True)
    #train and test on kfold
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


#comparing algorithms
plt.boxplot(results, tick_labels = names)
plt.title('Algorithm Comparison')
plt.show()




print('*******************************************************************************')
#make predictions on validation dataset

model = SVC(gamma = 'auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

#evaluate predictions
print('Accuracy score: ', accuracy_score(y_validation, predictions))
print('Confusion matrix: ')
print(confusion_matrix(y_validation, predictions))
print('Classification report: ')
print(classification_report(y_validation, predictions))

print('*******************************************************************************')


#save the trained model
joblib.dump(model, 'iris_model.pkl')
