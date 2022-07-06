import os
import pandas as pd
def add_teams():
    my_path = '/Users/kylegreen/codeup-data-science/time-series-exercises/riot_api_data/'
    _, _, filenames = next(os.walk(my_path), (None, None, []))
    #print(filenames)
    count= 0
    sum = 0

    for filename in filenames:
        if 'lobby' in filename:
            count+=1
            lobby_df = pd.read_csv(my_path+filename)
            lobby_df['is_blue_team'] = 0
            
            for col in lobby_df.columns:
                if 'Unnamed' in col:
                    lobby_df.drop(columns=col, inplace=True)
            # blue team
            lobby_df.iloc[0:5, lobby_df.columns.get_loc('is_blue_team')] = 1

            # red team
            lobby_df.iloc[5:, lobby_df.columns.get_loc('is_blue_team')] = 0
            sum += lobby_df.is_blue_team.sum()

            lobby_df.to_csv(my_path+filename, index=False)
