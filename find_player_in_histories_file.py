# find_player_in_histories_file
import os
import pandas as pd

def find_summoner_in_files(summoner_name):
    my_path = '/Users/kylegreen/codeup-data-science/time-series-exercises/riot_api_data/'
    _, _, filenames = next(os.walk(my_path), (None, None, []))
    #print(filenames)
    
    for file in filenames:
        if 'histories' in file:
            csv = pd.read_csv(my_path+file)
            bool = csv.summoner_name.str.contains(summoner_name)
            for b in bool:
                if b:
                    #print('found file')
                    return file
    print('No file found with summoner name')
