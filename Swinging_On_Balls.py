import pandas as pd
import numpy as np
import pickle
import BaseballUtils

model = pickle.load(open('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\swing_model.sav', 'rb'))
data = pd.read_csv('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\predictors.csv')


pitchers = pd.read_csv('C:\\Users\\Andrew Moss\\Documents\\Pitch_data_rangers_technical2.csv')
pitchers = pitchers.drop_duplicates()
pitchers = pitchers[pitchers['balls'] < 4]
pitchers = pitchers[pitchers['strikes'] < 3]
pitcherTable = BaseballUtils.getNamesFromIDs(pitchers)

pitchers = pd.merge(pitchers, pitcherTable, how='left', left_on='pitcher', right_on='pitcher')
data['pitcher'] = pitchers.loc[:, 'player_name_y'].values
data['zone'] = pitchers.loc[:, 'zone'].values
data['out_of_zone'] = np.where(data['zone'] > 9, 1, 0)
data = data[data['out_of_zone'] == 1]
data['Swing_prob'] = model.predict_proba(data.drop(['Unnamed: 0', 'pitcher', 'is_swing','zone','out_of_zone'],
                                                   axis=1))[:, 1]
data['SwOE'] = data['is_swing'] - data['Swing_prob']
data = data[data['pitcher'].notna()]
by_pitcher = data.groupby('pitcher').mean()
by_pitcher = by_pitcher.sort_values(by='SwOE', ascending=False)

by_pitcher.loc[:, 'SwOE'].\
    to_csv('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\SwOE_on_balls_by_pitcher.csv')
