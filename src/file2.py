# This notebook for run MLflow with dagshub remote access.

import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# This is only the extra code for remote on dagshub
# these things we have to copy from dagshub site when we connect this repo through github and use as usual. 
# I have to copy these and paste in correct place.

import dagshub
dagshub.init(repo_owner='Abhijit_Palai', repo_name='MLOPS-Experiments-with-MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Abhijit_Palai/MLOPS-Experiments-with-MLFlow.mlflow")

# -----------------------------------------------------------------------------------------------

# load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# train test split
x_train , x_test , y_train , y_test = train_test_split(X,y,test_size = 0.10 , random_state = 42)

# Define the params for RF model

max_depth = 10
n_estimators = 5

# mention your experiment below
mlflow.set_experiment('YT-MLOPS-EXP1')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth , n_estimators=n_estimators , random_state = 42)
    rf.fit(x_train , y_train)
    
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test , y_pred)
    
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    
    # Creating a confusion matrix log
    cm = confusion_matrix(y_test , y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm , annot=True , fmt = 'd',cmap = 'Blues',xticklabels=wine.target_names , yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # save plot
    plt.savefig('Confusion-Matrix.png') 
    
    # log artifacts using mlflow 
    mlflow.log_artifact('Confusion-Matrix.png')  
    mlflow.log_artifact(__file__)
    
    # Tags
    mlflow.set_tags({'Author':'Abhijit' , 'project':'wine_classification'})
    
    #Log the model
    mlflow.sklearn.log_model(rf , 'RandomForestClassifier')
    
    print(accuracy)