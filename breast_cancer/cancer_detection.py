# This model predicts whether the the tumour is benign(not breast cancer) or malignant(breast cancer) based off its characteristics. 
# Output 2 - benign, 4 - malignant

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


class CancerDetection:
    def __init__(self, datapath):
        self.dataset = datapath
    
    def logistic_regression(self):
        # Importing the self.dataset
        x = self.dataset.iloc[:, 1:-1].values
        y = self.dataset.iloc[:, -1].values

        # Splitting the data into train and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        
        # Training the model on training set
        classifier = LogisticRegression(random_state=0)
        classifier.fit(x_train, y_train)
        
        # Predicting the output
        y_pred = classifier.predict(x_test)
        # print(y_pred)

        # You can also predict by inputting your own values. Note: Predict method accepts 2d array
        # print(classifier.predict([[input 10 values of the cancer report here separated by commas]]))

        # Generating confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Computing accuracy with K-fold cross validation
        accuracy = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
        print(cm)
        print(f'Accuracy {accuracy.mean()*100:.2f} %')
        print(f'Standard deviation {accuracy.std()*100:.2f} %')
        # print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    dataset = pd.read_csv('E:/breast_cancer/Final Folder/Dataset/breast_cancer.csv')
    instance = CancerDetection(dataset)
    instance.logistic_regression()
