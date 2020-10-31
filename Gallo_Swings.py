import pandas as pd
import numpy as np
import pickle
import BaseballUtils
import plotly
import plotly.figure_factory as ff
from scipy.stats import binned_statistic_2d
import plotly.express as px
import plotly.graph_objects as go
import streamlit

data = pd.read_csv('C:\\Users\\Andrew Moss\\Downloads\\Gallo_2020.csv')
data['is_swing'] = 0

data = data[data['balls'] < 4]
data = data[data['strikes'] < 3]
strike_indices = np.flatnonzero(np.core.defchararray.find(data['description'].tolist(), 'called') != -1)
ball_indices = np.flatnonzero(np.core.defchararray.find(data['description'].tolist(), 'ball') != -1)
pitchout_indices = np.flatnonzero(np.core.defchararray.find(data['description'].tolist(), 'pitchout') != -1)
data = BaseballUtils.assignCountCluster(data)
Count_cluster = data['count_cluster']

called_indices = np.concatenate((ball_indices, strike_indices, pitchout_indices), axis=0)
data['is_swing'] = np.where(np.isin(data.index, called_indices), 0, 1)
data['Swing_percent'] = np.full(len(data), sum(data['is_swing'])/len(data))
top = sum(data['sz_top'])/len(data)
bottom = sum(data['sz_bot'])/len(data)




model = pickle.load(open('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\swing_model.sav', 'rb'))

cols_when_model_builds = model.get_booster().feature_names
data = pd.get_dummies(data=data, columns=['pitcher_side', 'batter_side', 'outs_when_up', 'pitch_type', 'count_cluster'])
data['batter_side_R'] = 0
data['pitch_type_CS'] = 0
data['pitch_type_FO'] = 0
data['pitch_type_KN'] = 0
is_swing = data['is_swing']
data = data.loc[:, cols_when_model_builds]




data['Class_prob'] = model.predict_proba(data)[:, 1]
data = data.loc[:, ['plate_x', 'plate_z', 'Class_prob']]
data['is_swing'] = is_swing
data['SwOE'] = data['is_swing'] - data['Class_prob']
data['count_cluster'] = Count_cluster

biny = np.arange(-3, 3.33, .33)
binx = np.arange(-2, 6.33, .33)
heat_list = []
all_yet = False
dummy_list = [0]
for count in [0, 1, 2, 3, 4, 5, 6, 7]:
    if all_yet == False:
        count_data = data
    else:
        print(count)
        count_data = data[data['count_cluster'] == count]
        count_data = count_data.reset_index()
    ret = binned_statistic_2d(count_data['plate_z'], count_data['plate_x'], None, 'count', bins=[binx, biny],
                              expand_binnumbers=True)
    averages = np.zeros([-1+len(binx), -1+len(biny)])
    for i in range(len(count_data)):
        m = ret.binnumber[0][i] - 1
        n = ret.binnumber[1][i] - 1
        averages[m][n] = averages[m][n] + count_data['SwOE'][i]
    averages = np.divide(averages, ret.statistic)
    averages = np.nan_to_num(averages, copy=True, nan=0)
    heat_list.append(averages)
    all_yet = True

figList = []
count = 0
text = ""

for item in heat_list:
    if count == 0:
        text = "Joey Gallo (L) Swings Over Expectation, All Counts 2020"
    elif count == 1:
        text = 'Joey Gallo (L) Swing Rates Over Expectation for Count Cluster ' + str(count) + ": " + "3-0 Counts, 2020"
    elif count == 2:
        text = 'Joey Gallo (L) Swing Rates Over Expectation for Count Cluster ' + str(count) + ": " + "0-0 Counts, 2020"
    elif count == 3:
        text = 'Joey Gallo (L) Swing Rates Over Expectation for Count Cluster ' + str(count) + ": " + "1-0, 2-0, 0-1 " \
                                                                                                  "Counts, 2020"
    elif count == 4:
        text = 'Joey Gallo (L) Swing Rates Over Expectation for Count Cluster ' + str(count) + ": " + "0-2, 1-1 Counts," \
                                                                                                  " 2020"
    elif count == 5:
        text = 'Joey Gallo (L) Swing Rates Over Expectation for Count Cluster ' + str(count) + ": " + "3-1, 2-1, 1-2 " \
                                                                                                  "Counts, 2020"
    elif count == 6:
        text = 'Joey Gallo (L) Swing Rates Over Expectation for Count Cluster ' + str(count) + ": " + "2-2 Counts, 2020"
    else:
        text = 'Joey Gallo (L) Swing Rates Over Expectation for Count Cluster ' + str(count) + ": " + "3-2 Counts, 2020"
    fig = px.imshow(item, x=biny[1:], y=binx[1:],
                    title=text,
                    labels=dict(x="plate_x", y="plate_z"),
                    zmax=.8, zmin=-.8, color_continuous_scale=px.colors.sequential.Rainbow)
    fig.update_yaxes(autorange=True)
    fig.add_shape(type="rect",
                  xref="x", yref="y",
                  x0=-17/12, y0=top, x1=bottom, y1=17/12,
                  line=dict(
                      color="Black",
                      width=3,
                  ),
                  )
    figList.append(fig.data[0])
    count = count + 1



# for fig in figList:
#     fig.show()
chart = streamlit.plotly_chart(figList, use_container_width=False, sharing='streamlit')




