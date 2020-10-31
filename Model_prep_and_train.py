import pandas as pd
import numpy as np
import BaseballUtils
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from xgboost import XGBClassifier
import pickle




fulldata = pd.read_csv('C:\\Users\\Andrew Moss\\Documents\\Pitch_data_rangers_technical2.csv')
data = fulldata.drop(['game_date', 'pitcher', 'events', 'zone', 'game_pk',
                      'game_type', 'type', 'fielder_2', 'pitch_number', 'inning_topbot'], axis=1)

data = data.drop_duplicates()
data['is_swing'] = 0

data = data[data['balls'] < 4]
data = data[data['strikes'] < 3]
strike_indices = np.flatnonzero(np.core.defchararray.find(data['description'].tolist(), 'called') != -1)
ball_indices = np.flatnonzero(np.core.defchararray.find(data['description'].tolist(), 'ball') != -1)
pitchout_indices = np.flatnonzero(np.core.defchararray.find(data['description'].tolist(), 'pitchout') != -1)

called_indices = np.concatenate((ball_indices, strike_indices, pitchout_indices), axis=0)
data['is_swing'] = np.where(np.isin(data.index, called_indices), 0, 1)
data = BaseballUtils.assignCountCluster(data)


by_cat = data.groupby('count_cluster').mean()
plt.plot(by_cat.index, by_cat['is_swing'])
plt.ylabel('Swing Percentage')
plt.title('Swing percentage by Count Cluster (Hierarchical)')
plt.savefig('countCluster')

corr = data.corr()

data = pd.get_dummies(data=data, columns=['pitcher_side', 'batter_side', 'outs_when_up', 'pitch_type', 'count_cluster'])
data['Swing_percent'] = 0
playerSwing = data.groupby(['batter']).mean()

for index, row in playerSwing.iterrows():
    print(index)
    data['Swing_percent'] = np.where(data['batter'] == index, row['is_swing'], data['Swing_percent'])

data = data.drop(['description', 'batter', 'player_name', 'balls', 'strikes', 'ax', 'az', 'effective_speed',
                  'sz_bot', 'inning', 'vy0'], axis=1)


headers = data.columns

imputer = KNNImputer(n_neighbors=100, weights='distance')
data = imputer.fit_transform(data)
data = pd.DataFrame(data)
data.columns = headers

predframe = data.drop(['is_swing'], axis=1)
y = data.loc[:, 'is_swing']

model_kfold = XGBClassifier(objective='binary:logistic')
kfold = StratifiedKFold(n_splits=10)

results_kfold = model_selection.cross_val_score(model_kfold, predframe, y, scoring='balanced_accuracy',
                                                cv=kfold, n_jobs=-1)
print("Accuracy full: %.2f%%" % (results_kfold.mean()*100.0))

model_kfold.fit(predframe, y)
pickle.dump(model_kfold, open('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\swing_model.sav', 'wb'))

data.to_csv('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\predictors.csv')
