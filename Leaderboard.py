import pandas as pd

SWOE = pd.read_csv('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\SwOE_on_balls_by_pitcher.csv')
SWUE = pd.read_csv('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\SwUE_on_strikes_by_pitcher.csv')

Leaderboard = pd.merge(SWOE, SWUE, how='inner', on='pitcher')
Leaderboard['Deception_Over_Expectation'] = Leaderboard['SwOE'] + Leaderboard['SwUE']
Leaderboard = Leaderboard.sort_values('Deception_Over_Expectation', ascending=False)

Leaderboard.to_csv('C:\\Users\\Andrew Moss\\PycharmProjects\\Rangers_Survey_2\\Leaderboard.csv')
