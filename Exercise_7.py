from sklearn.datasets import load_iris
iris=load_iris()                                                                        
X=iris.data     
y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.tree import DecisionTreeClassifier
d=DecisionTreeClassifier()
d.fit(X_train,y_train)
y_pred=d.predict(X_test)
print("\nPredicted : ",y_pred)
print("Original    : ",y_test)


from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
print("\nAccuracy : ",acc)

from sklearn import tree
import graphviz
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(d, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)


sample=[[2,3,1,2]]
y_pred_2=d.predict(sample)
print("\nSample Value      :",sample)
print("Sample prediction : ",iris.target_names[y_pred_2])
