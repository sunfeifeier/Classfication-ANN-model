# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:20:00 2017
https://gist.github.com/HasseIona/4bcaf9f95ae828e056d5210a2ea07f88
https://medium.com/@williamkoehrsen/deep-neural-network-classifier-32c12ff46b6c
learning rate： 太大，震荡不收敛，太小，收敛慢。
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
grid search
https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search

#optimize the hidden layers number and size
https://machinelearningmastery.com/exploratory-configuration-multilayer-perceptron-network-time-series-forecasting/
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/


@author: fei
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
import keras
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from math import sqrt
from numpy import array
from keras import optimizers 
from keras.optimizers import RMSprop,SGD,Adam
from numpy import concatenate
import seaborn as sns
from scipy import stats, integrate
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = "Times New Roman"
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["figure.figsize"] = [12, 6]


#df=read_csv("Phoneix_Finalclean.csv",index_col='new_UTC')
df = read_csv("Phoneix_Finalclean.csv")
aa=df.Conditions_Name.value_counts()
ax=aa.plot(x='Conditions_Name', y='Amount',kind='bar',color="blue", figsize=(15,8),fontsize=16)
#plt.title("Twelve years' weather condition summary",size=30)
#ax.set_title("2004-2016 Phoenix weather condition summary",size=30)
ax.set_xlabel('Weather Condition',size=20) 
ax.set_ylabel('Total Amount/hour',size=20)                    
plt.show()


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_d =df.loc[:,["Sea_Level_PressureIn_N","Humidity_N","Dew_PointF_N","Wind_Speed_mps","Temperature_C_N"]]
scaled = scaler.fit_transform(scaled_d)
scaled = DataFrame(scaled)
scaled.columns = ["Sea_Level_PressureIn_N","Humidity_N","Dew_PointF_N","Wind_Speed_mps","Temperature_C_N"]
x = scaled.join(df.loc[:,["Hour","Conditions_Name"]])
x = x.loc[x['Conditions_Name'].isin(["Clear","Mostly Cloudy","Partly Cloudy","Scattered Clouds",'Overcast'])]
x = x.dropna()
x.isnull().sum()
count = x.Conditions_Name.value_counts()
print(count)

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
x = encode(x, 'Hour', 23)
x=x.drop(["Hour"],axis=1)
ax = x.plot.scatter('Hour_sin', 'Hour_cos').set_aspect('equal')
"""
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg 

rnn_data=series_to_supervised(x,3,1)
rnn_data=rnn_data.drop(["var7(t-1)","var8(t-1)",
     "var7(t-2)","var8(t-2)","var7(t-3)","var8(t-3)"],axis=1)

rnn_data.rename(columns={'var1(t)': 'Sea_Level_PressureIn_N(t)', 'var2(t)': "Humidity_N(t)",
  'var3(t)': 'Dew_PointF_N(t)', 'var4(t)':"Wind_Speed_mps(t)",'var5(t)': "Temperature_C_N(t)", 
  'var6(t)':"Conditions_Name(t)",
  'var1(t-1)': 'Sea_Level_PressureIn_N(t-1)', 'var2(t-1)': "Humidity_N(t-1)",
  'var3(t-1)':'Dew_PointF_N(t-1)', 'var4(t-1)':"Wind_Speed_mps(t-1)",'var5(t-1)': "Temperature_C_N(t-1)", 
  'var6(t-1)':'Conditions_Name(t-1)',
  'var1(t-2)': 'Sea_Level_PressureIn_N(t-2)', 'var2(t-2)': "Humidity_N(t-2)",
  'var3(t-2)': 'Dew_PointF_N(t-2)', 'var4(t-2)':"Wind_Speed_mps(t-2)",'var5(t-2)': "Temperature_C_N(t-2)", 
  'var6(t-2)':"Conditions_Name(t-2)",
  'var1(t-3)': 'Sea_Level_PressureIn_N(t-3)', "var2(t-3)":"Humidity_N(t-3)",
  'var3(t-3)': 'Dew_PointF_N(t-3)', 'var4(t-3)':"Wind_Speed_mps(t-3)",'var5(t-3)': "Temperature_C_N(t-3)", 
  'var6(t-3)':"Conditions_Name(t-3)",
   "var7(t)":'Hour_sin', "var8(t)":'Hour_cos'}, inplace=True)

encoder = LabelEncoder()
rnn_data["Conditions_Name(t-1)"]=encoder.fit_transform(rnn_data["Conditions_Name(t-1)"].astype("str"))
rnn_data["Conditions_Name(t-2)"]=encoder.fit_transform(rnn_data["Conditions_Name(t-2)"].astype("str"))
rnn_data["Conditions_Name(t-3)"]=encoder.fit_transform(rnn_data["Conditions_Name(t-3)"].astype("str"))

aa=pd.DataFrame(rnn_data)
#aa.to_csv("HMM_data".csv)
# sclaing condition#
rnn_data[["Conditions_Name(t-1)", "Conditions_Name(t-2)","Conditions_Name(t-3)"]] = scaler.fit_transform(rnn_data[["Conditions_Name(t-1)", "Conditions_Name(t-2)","Conditions_Name(t-3)"]])   
rnn_data.to_csv("rnn_data.csv")
"""

"""
feature selection
https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/
"""
#Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
X_ = rnn_data.drop(["Conditions_Name(t)"],axis=1)
X_=X_.astype(float)
Y_ = rnn_data["Conditions_Name(t)"].values
encoder = LabelEncoder()
encoder.fit(Y_)
encoded_Y = encoder.transform(Y_)
model = LogisticRegression(multi_class="multinomial",solver='lbfgs')
rfe = RFE(model, 18)
fit = rfe.fit(X_, encoded_Y)
print(("Num Features: %d") % fit.n_features_)
print(("Selected Features: %s") % fit.support_)
print(("Feature Ranking: %s") % fit.ranking_)

# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_, encoded_Y)
# display the relative importance of each attribute
print(model.feature_importances_)

######################

_ = sns.swarmplot(x='Sea_Level_PressureIn_N(t)', y='Conditions_Name(t)', data=x)
_ = plt.xlabel('species')
_ = plt.ylabel('petal lengths.')
plt.show()

"""
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas','class']
dataframe = read_csv(url, names=names)
"""

def encode_text_index(df, name):
    le = LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90)

    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#OD_2017=OD_2017.drop(columns=['Temperature_C']) 

"""
dataset1=rnn_data.loc[:,['Sea_Level_PressureIn_N(t-3)',
       'Dew_PointF_N(t-3)', 'Wind_Speed_mps(t-3)', 'Temperature_C_N(t-3)',"Conditions_Name(t-3)",
       'Sea_Level_PressureIn_N(t-2)','Humidity_N(t-2)', 'Dew_PointF_N(t-2)', 
       'Wind_Speed_mps(t-2)',"Conditions_Name(t-2)",
       'Sea_Level_PressureIn_N(t-1)', 'Humidity_N(t-1)',
       'Dew_PointF_N(t-1)', 'Wind_Speed_mps(t-1)', "Conditions_Name(t-1)"
       'Sea_Level_PressureIn_N(t)',
       'Humidity_N(t)', 'Dew_PointF_N(t)','Wind_Speed_mps(t)',
       'Temperature_C_N(t)', 'Hour_sin', 'Hour_cos',"Conditions_Name(t)"]].values

"""

###### parameret selection
rnn_data=read_csv("rnn_data.csv")
X_train=rnn_data.loc[:,[
       'Sea_Level_PressureIn_N(t)','Sea_Level_PressureIn_N(t-1)',
       'Humidity_N(t)', 'Dew_PointF_N(t)','Wind_Speed_mps(t)',
       'Temperature_C_N(t)', 'Hour_sin', 'Hour_cos','Temperature_C_N(t-1)',
       'Dew_PointF_N(t-1)',
       "Conditions_Name(t-1)"]]
X_train=X_train.values
Y=rnn_data["Conditions_Name(t)"]
encoder = LabelEncoder()
encoder.fit(Y)
Y_train = encoder.transform(Y)

X11=X_train[0:200]
plt.plot("Conditions_Name(t-1)",c="b",data=X11)
plt.plot("Temperature_C_N(t)",c="r",data=X11)
plt.plot("Dew_PointF_N(t)",c="g",data=X11)
plt.show()
n_class=4
input_shape=(8,)
# select optimizer and layers
def create_model(dense_layers=[10],activation='relu', optimizer="adam"):
# create model
    model = Sequential()
    for index, lsize in enumerate(dense_layers):
        # Input Layer - includes the input_shape
        if index == 0:
            model.add(Dense(lsize,
                            activation=activation,
                            input_shape=(8,)))
        else:
            model.add(Dense(lsize,
                            activation=activation))
    model.add(Dense(4, activation="softmax"))
    # Compile model
    model.compile(loss="categorical_crossentropy", 
                  optimizer=optimizer, metrics=["accuracy"])   
    return model

model = KerasClassifier(build_fn=create_model, 
                        epochs=10, 
                        batch_size=2, 
                        verbose=1)
#define the grid search parameters
param_grid = {'dense_layers': [[100],[200],[300],[300,300]],
              'activation':['relu','tanh'],
              'optimizer':('rmsprop','adam'),
              'epochs':[10,20,30],
              'batch_size':[2,4]}

grid = GridSearchCV(model, 
                    param_grid=param_grid, 
                    return_train_score=True,
                    scoring=['precision_macro','recall_macro','f1_macro'],
                    refit='precision_macro')
           
grid_result = grid.fit(X_train, Y_train)
print('Parameters of the best model: ')
print(grid_results.best_params_)



### the model with best parameter
dataset1=rnn_data.loc[:,[
       'Sea_Level_PressureIn_N(t)',
       'Humidity_N(t)', 'Dew_PointF_N(t)','Wind_Speed_mps(t)',
       'Temperature_C_N(t)', "Conditions_Name(t-1)","Conditions_Name(t-2)"，
       "Conditions_Name(t)"]]
#'Dew_PointF_N(t-1)',

Conditions_Name= encode_text_index(dataset1,"Conditions_Name(t)")
X,Y = to_xy(dataset1,"Conditions_Name(t)")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
## Capture the best params
params = grid_results.best_params_

## create the model with the best params found
model = create_model(dense_layers=params['dense_layers'],
                     activation=params['activation'],
                     optimizer=params['optimizer'])

## Then train it and display the results
history = model.fit(X_train,
                    Y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose = 0)

model.summary()
pred = model.predict(X_test)
real_y = np.argmax(Y_test,axis=1)
pred_y = np.argmax(pred,axis=1) 
# Compute confusion matrix
cm = confusion_matrix(real_y, pred_y)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, Conditions_Name)

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, Conditions_Name, title='Normalized confusion matrix')
plt.show()


#### using the best parameter into model
dataset1=rnn_data.loc[:,[
       'Sea_Level_PressureIn_N(t-1)',
       'Humidity_N(t-1)',
       'Dew_PointF_N(t-1)',
       'Wind_Speed_mps(t-1)',
       'Temperature_C_N(t-1)',
       "Hour_sin","Hour_cos",
       "Conditions_Name(t-1)",
       "Conditions_Name(t)"]]

Conditions_Name= encode_text_index(dataset1,"Conditions_Name(t)")
X,Y = to_xy(dataset1,"Conditions_Name(t)")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
n_epochs=20
n_batch =1

model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, kernel_regularizer=regularizers.l2(0.01),
                  activity_regularizer=regularizers.l1(0.01),
                  activation='relu'))
#model.add(LSTM(300,return_sequences=True,activation='tanh',stateful=True,
#               batch_input_shape=(n_batch, time_step, X_train.shape[2])))
#model.add(LSTM(100, input_shape=(time_step,n_features)))
model.add(Dense(Y_train.shape[1], activation='softmax'))
# Compile model
adam=keras.optimizers.Adam(lr=0.01, beta_1=0.99, beta_2=0.99, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath="best_weights_mod5.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),callbacks=callbacks_list,verbose=1,epochs=n_epochs,batch_size=n_batch)
###plot loss

acc_mod1=[]
val_acc_mod1=[]
acc_mod1.append(history.history["acc"])
val_acc_mod1.append(history.history["val_acc"])
acc_mod1=pd.DataFrame(np.array(acc_mod1).reshape(n_epochs,1))
val_acc_mod1=pd.DataFrame(np.array(val_acc_mod1).reshape(n_epochs,1))
acc_mod1.rename(columns={acc_mod1.columns[0]:"Accuracy"},inplace=True)
val_acc_mod1.rename(columns={val_acc_mod1.columns[0]:"Val_Accuracy"},inplace=True)

Acc_mod=acc_mod1.join(val_acc_mod1)
Acc_mod.to_csv("trainresult_mod_5.csv")


draftdata=read_csv("Training Result.csv")
#plt.plot('mod1_loss', label='MSE_Train',c="r",data=draftdata)
plt.plot('mod2_loss', label='MSE_Train',c="b",data=draftdata)
plt.plot('mod3_loss', label='MSE_Train',c="gray",data=draftdata)
plt.plot('mod4_loss', label='MSE_Train',c="green",data=draftdata)
plt.plot('mod5_loss', label='MSE_Train',c="m",data=draftdata)
plt.plot('mod6_loss', label='MSE_Train',c="y",data=draftdata)
plt.plot('mod7_loss', label='MSE_Train',c="b",data=draftdata)
plt.plot('mod8_loss', label='MSE_Train',c="c",data=draftdata)
#plt.plot('mod9_loss', label='MSE_Train',c="k",data=draftdata)
#plt.title('Model Train loss',size=16)
plt.legend([ 'mod2','mod3','mod4','mod5',"mod6","mod7","mod8"], loc='upper right')
plt.ylabel('loss',fontsize=12)
plt.xlabel('epoch',fontsize=12)
plt.show()

plt.plot('mod8_loss', label='MSE_Train',c="c",data=draftdata)
plt.plot('mod1_loss', label='MSE_Train',c="r",data=draftdata)
plt.legend([ 'mod1',"mod8"], loc='upper right')
plt.ylabel('loss',fontsize=12)
plt.xlabel('epoch',fontsize=12)
plt.show()


#plt.plot('mod1_Accuracy', c="r",data=draftdata)
plt.plot('mod2_Accuracy', c="b",data=draftdata)
plt.plot('mod3_Accuracy', c="gray",data=draftdata)
plt.plot('mod4_Accuracy', c="green",data=draftdata)
plt.plot('mod5_Accuracy', c="m",data=draftdata)
plt.plot('mod6_Accuracy', label='MSE_Train',c="y",data=draftdata)
plt.plot('mod7_Accuracy', label='MSE_Train',c="b",data=draftdata)
plt.plot('mod8_Accuracy', label='MSE_Train',c="c",data=draftdata)
#plt.plot('mod9_loss', label='MSE_Train',c="k",data=draftdata)
#plt.title('Model Train Accuracy',size=20)
plt.legend(['mod8','mod3','mod4','mod5',"mod6","mod7","mod2"], loc='upper right')
plt.ylabel('Auccuracy')
plt.xlabel('epoch')
plt.show()


plt.plot('mod1_Accuracy', c="r",data=draftdata)
plt.plot('mod2_Accuracy', label='MSE_Train',c="b",data=draftdata)
plt.legend(['mod1',"mod8"], loc='upper right')
plt.ylabel('Auccuracy')
plt.xlabel('epoch')
plt.show()


with plt.style.context('Solarize_Light2'):
    plt.plot('mod1_loss', data=draftdata)
    plt.plot('mod2_loss', data=draftdata)
    plt.plot('mod3_loss', data=draftdata)
    plt.plot('mod4_loss', data=draftdata)
    plt.plot('mod5_loss', data=draftdata)
    plt.plot('mod6_loss', data=draftdata)
    plt.plot('mod8_loss', data=draftdata)
    plt.plot('mod9_loss', data=draftdata)
    plt.title('Model Train loss',size=20)
    plt.legend(['mod1', 'mod2','mod3','mod4','mod5',"mod6","mod7","mod8"], loc='upper right')
    plt.ylabel('loss',fontsize=18)
    plt.xlabel('epoch',fontsize=18)
plt.show()





###predicted
#model.load_weights('best_weights_mod3.hdf5') # load weights from best model
loss, acc = model.evaluate(X_test,Y_test)
pred = model.predict(X_test)

from sklearn import metrics
pred_y = np.argmax(pred,axis=1)
real_y = np.argmax(Y_test,axis=1)
score=metrics.accuracy_score(real_y,pred_y)
score=metrics.accuracy_score(real_y,index)
print("accuracy score:{}".format(score))

#print(metrics.log_losspre())
a=encoder.inverse_transform(pred_y)
a=pd.DataFrame(a)
real_y = np.argmax(Y_test,axis=1)
pred_y = np.argmax(pred,axis=1) 
index_chosing=read_csv('index_choosing_1.csv')
index=index_chosing["Final"]


# Compute confusion matrix
cm = confusion_matrix(real_y, pred_y) ## Real test result
cm = confusion_matrix(real_y, index)  ## clear day is greater 50%
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, Conditions_Name)

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, Conditions_Name, title='Normalized confusion matrix')
plt.show()


a=[]
for i in range(len(pred)):
    if pred.loc[i,"Clear"]>=0.5:
        a=0
    else:
        a=
    for i in range(len(df)) : 
  print(df.loc[i, "Name"], df.loc[i, "Age"]) 

#####################################################

#### PCA分析######
X = x.loc[100000:,["Month","Day","Hour","Sea_Level_PressureIn_N","Humidity_N","Dew_PointF_N","Wind_Speed_mps","Temperature_C_N"]]
Y= x.loc[100000:,["Conditions_Name"]]

Y=pd.Categorical(Y)

my_color=Y.cat.codes
my_color=['r', 'g', 'm']


scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)

principalComponents = pca.fit_transform(X)
result = DataFrame(data = principalComponents
             , columns = ['PCA%i' % i for i in range(3)], index=X.index)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=my_color, cmap="Conditions_Name", s=60)

# make simple, bare axis lines through space:
xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on the iris data set")
#plt.show()



principalDf = DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = concat([principalDf, x[["Conditions_Name"]]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ["Clear","Scattered Clouds","Mostly Cloudy","Partly Cloudy"]

colors = ['r', 'g', 'm',"y","k","c","b"]
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Conditions_Name'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

############################################

##save and load model###########
import os
from keras.models import model_from_json

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weight("model.h5")


from keras.model import load_model
json_file=open("model.json","r")
loaded_model_json=json_model.read()
json_file.close()
loaded_model.load_weights("model.h5")

pred=loaded_model.predict(input_data)
load_model.compile()
score=load_model.evaluate(input,output)

################################



#############333
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dataset1 = x.loc[:,["Day_cos","Sea_Level_PressureIn_N","Humidity_N","Dew_PointF_N","Wind_Speed_mps","Temperature_C_N","Conditions_Name"]]
dataset=dataset1.values
X = dataset[:,4:8].astype(float)
Y = dataset[:,8]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
X_train = X[0:96432]
Y_train = dummy_y[0:96432]
X_test = X[96432:105192]
Y_test = dummy_y[96432:105192]
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
	model.add(Dense(Y_train.shape[1], activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#	moni
  return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=4, verbose=0)
#train_test_split(X, dummy_y, test_size=0.2, random_state=seed)
estimator.fit(X_train, Y_train)

history = estimator.fit(X_train, Y_train)
history.history['loss']
history.history["acc"]
predictions = estimator.predict(X_test)
a=encoder.inverse_transform(predictions)
pd.DataFrame(a).to_csv("Boston_predict2015_weather.csv",header=["Predict_Condition"])



###real value:
real=x.loc[:,["Year","Conditions_Name"]]
cond=real.groupby("Year")
cond1=cond.get_group(2015)
cond2=cond1.groupby("Conditions_Name")
real_cond=cond2.agg(np.size)
real_cond=real_cond.reset_index()
real_cond = real_cond.rename(columns={real_cond.columns[1]: "Amount_real"})
real_cond.index = real_cond.index + 1
#


##predict values
predict = read_csv(".csv") 
pred_1=predict.groupby("Predict_Condition")
pred_cond=pred_1.agg(np.size)
pred_cond=pred_cond.reset_index()
pred_cond.index = pred_cond.index + 1
pred_cond = pred_cond.rename(columns={pred_cond.columns[1]: "Amount_pred"})
b=real_cond.join(pred_cond)
pd.DataFrame(b).to_csv(".csv",index=False)



#####################
#mapping location
import folium
phone_map = folium.Map()

# Top three smart phone companies by market share in 2016
companies = [
    {'loc': [37.4970,  127.0266], 'label': 'Samsung: 20.5%'},
    {'loc': [37.3318, -122.0311], 'label': 'Apple: 14.4%'},
    {'loc': [22.5431,  114.0579], 'label': 'Huawei: 8.9%'}] 

# Adding markers to the map
for company in companies:
    marker = folium.Marker(location=company['loc'], popup=company['label'])
    marker.add_to(phone_map)

# The last object in the cell always gets shown in the notebook
phone_map
###################

#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
#Model selection KERAS




import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))
# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
history=model.fit(data, one_hot_labels, epochs=10, batch_size=1)
history.history['acc']
#history.history['val_acc']
score = model.evaluate(data, one_hot_labels, batch_size=1)

#https://gist.github.com/HasseIona/4bcaf9f95ae828e056d5210a2ea07f88   
 
 # create model
#https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

