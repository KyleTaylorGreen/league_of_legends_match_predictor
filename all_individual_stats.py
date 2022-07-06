import os
import pandas as pd
import numpy as np
import datetime
import add_team_data
import modeling as md
import split
import time
import requests

my_path = '/Users/kylegreen/codeup-data-science/time-series-exercises/riot_api_data/'
_, _, filenames = next(os.walk(my_path), (None, None, []))

def create_player_histories():
    my_path = '/Users/kylegreen/codeup-data-science/time-series-exercises/riot_api_data/'
    _, _, filenames = next(os.walk(my_path), (None, None, []))
    
    if os.path.isfile(my_path + 'all_stats_histories.csv'):
        return pd.read_csv(my_path+ 'all_stats_histories.csv')
    
    else:
        total_df = pd.DataFrame()
        for filename in filenames:
            if 'histories' in filename:
                histories_df = pd.read_csv(my_path+filename)
                
                for col in histories_df.columns:
                    if 'Unnamed' in col:
                        histories_df.drop(columns=col, inplace=True)
                
                total_df = pd.concat([total_df, histories_df])
   
    total_df= total_df.drop_duplicates()
    total_df.to_csv(my_path+'all_stats_histories.csv', index=False)
    return total_df

def update_all_stats_histories():

    folder = '/Users/kylegreen/codeup-data-science/time-series-exercises/riot_api_data/'
    _, _, filenames = next(os.walk(folder), (None, None, []))
    histories = create_player_histories()
    history_filenames = [file for file in filenames if file != 'all_stats_histories.csv' and 'lobby' not in file]

    list_of_histories = [histories]

    new_history_files = [file for file in history_filenames if np.isin(file[:-14], histories.game_id.unique())]
    for file in new_history_files:
        list_of_histories.append(pd.read_csv(folder+file))

    histories = pd.concat(list_of_histories)
    histories.dropna(inplace=True)
    
    histories.to_csv(folder+'all_stats_histories.csv', index=False)
    
    return histories

def drop_duplicates():
    total_df = pd.read_csv('all_stats_histories.csv')
    total_df = total_df.drop_duplicates()
    total_df.to_csv(my_path+'all_stats_histories.csv', index=False)

    #print(total_df)

def read_stats(filename):
    return pd.read_csv(filename)

def time_columns(filename):
    stats = read_stats(f'/Users/kylegreen/codeup-data-science/time-series-exercises/riot_api_data/{filename}')
    columns= [col for col in stats.columns]
    #stats.game_end = stats.game_end.fillna(0)
    #print(stats.game_end)
    stats['game_end'] = stats.apply(lambda x: x.game_start + (x.game_duration * 1000), axis=1)
   # stats['game_duration'] = stats.apply(lambda x: x.game_end - x.game_start, axis=1)
    stats = stats.sort_values(by=['summoner_name', 'game_start'], ascending=False)
    #print(pd.Timedelta(stats.game_end.iloc[1,] - stats.shift(1).game_start.iloc[1,]).seconds/60 < 45)
    #stats['is_session'] = stats.apply(lambda x: 1 if pd.Timedelta(x.game_end - x.shift(-1).game_start).seconds/60 < 45 and x.summoner_name == x.shift(1).summoner_name else 0, axis=1)
    stats['time_between_games'] = (stats['game_start'] - stats.shift(-1)['game_end']) / 1000 / 60
    another_df = stats.shift(-1)
    #print(another_df)
    stats['same_player_as_prev'] = stats['summoner_name'] == another_df['summoner_name']
    #print(stats.time_between_games)
    #print(stats.same_player_as_prev)
    stats['is_session'] = stats.apply(lambda x: True if x.time_between_games < 45 and x.same_player_as_prev == True else False, axis=1)
    #print(stats.is_session)
    columns.append('is_session')
    columns.append('time_between_games')
    columns.append('game_end')

    def make_session_counts(df):
        truth_counter= 1
        session_series = []
        session_cnt = 0
        sessions = [[]]
        wins_losses = []
        copy= [[]]

        win_counter = 0
        loss_counter = 0
        num_of_wins_in_session = 1
        
        for i, x in enumerate(df.is_session):
            if x == True:
                if df.win.iloc[i]:
                    win_counter +=1
                else:
                    loss_counter +=1
                truth_counter += 1
                sessions[session_cnt].append(i)
            else:
                for k in range(truth_counter, 0, -1):
                    session_series.append(k)
                
                truth_counter = 1
                sessions[session_cnt].append(i)
                sessions.append([])
                session_cnt+=1

        copy = sessions[:][:]
        counter = 0
        for l, session in enumerate(sessions):
            copy[l] = np.array(copy[l])

            for k, game in enumerate(session):
                #print(counter)
                #print(df.win.iloc[counter])
                if df.win.iloc[counter]:
                    copy[l][k] = 1
                else:
                    copy[l][k] = 0
                counter+=1
            copy[l] = np.flip(copy[l])
            copy[l] = np.cumsum(copy[l])
            copy[l] = np.flip(copy[l])

        copy = copy[:-1]
        final_wins = [elem for elements in copy for elem in elements]
        #print(final_wins)
        #print(session_series)
        #print(len(final_wins))

        return session_series, final_wins

    total_df = stats[columns]
    total_df = total_df.reset_index()

    our_session, final_wins = make_session_counts(total_df)
    #print(len(our_session))
    total_df['session_count'] = our_session
    total_df['session_wins']  = final_wins
    total_df['session_losses']= total_df.session_count - total_df.session_wins

    return total_df

def session_games_lookup(lobby_df_x, stats_df):
    if lobby_df_x.previous_game_id not in ['0', 0, np.NaN] and lobby_df_x.is_session:
        
        if stats_df.loc[(stats_df.summoner_name == lobby_df_x.summonerName) & (stats_df.game_id == lobby_df_x.previous_game_id)].session_count.size > 0:    
            return stats_df.loc[(stats_df.summoner_name == lobby_df_x.summonerName) & (stats_df.game_id == lobby_df_x.previous_game_id)].session_count.values.max()
        else:
            return 0
    else:
        return 0
    
def session_wins_lookup(lobby_df_x, stats_df):
    if lobby_df_x.previous_game_id not in ['0', 0, np.NaN] and lobby_df_x.is_session:
        #print(lobby_df_x.summonerName)

        if stats_df.loc[(stats_df.summoner_name == lobby_df_x.summonerName) & (stats_df.game_id == lobby_df_x.previous_game_id)].session_wins.values.size > 0:    
            return stats_df.loc[(stats_df.summoner_name == lobby_df_x.summonerName) & (stats_df.game_id == lobby_df_x.previous_game_id)].session_wins.values.max()
        else:
            return 0
    else:
        return 0
    
        
def session_losses_lookup(lobby_df_x, stats_df):
    
    if lobby_df_x.previous_game_id not in ['0', 0, np.NaN] and lobby_df_x.is_session:
        #print(lobby_df_x.summonerName)
        
            #if stats_df.loc[(stats_df.summoner_name == lobby_df_x.summonerName) & (stats_df.game_id == lobby_df_x.previous_game_id)].session_losses.values[0:    
            return stats_df.loc[(stats_df.summoner_name == lobby_df_x.summonerName) & (stats_df.game_id == lobby_df_x.previous_game_id)].session_losses.values[0]
    else:
        return 0

def find_missing_role_and_insert(lobby_df):
    
    missing_role = lobby_df.team_pos.value_counts().idxmin()
    lobby_df.team_pos = lobby_df.team_pos.fillna(missing_role)
    
    return lobby_df


def create_lobby_features(folder, lobby_filename): 
    """creates aggregate lobby stats up to time of game for each player,
    returns a lobby dataframe with stats attached"""
    stats_df = create_player_histories()
    stats_df = stats_df.dropna()
    add_team_data.add_teams()   
    
    path = folder+lobby_filename
    lobby_df = pd.read_csv(path) 
    
    
    stats_df = time_columns('all_stats_histories.csv')
    
    
    lobby_df['session_games'] = lobby_df.apply(lambda x: session_games_lookup(x,stats_df) if x.is_session else 0, axis=1)
    lobby_df['session_wins'] = lobby_df.apply(lambda x: session_wins_lookup(x,stats_df) if x.is_session else 0, axis=1)
    lobby_df['session_losses'] = lobby_df['session_games'] - lobby_df['session_wins']
    
    
    lobby_df['session_WL_ratio'] = lobby_df.apply(lambda x: x.session_wins / x.session_games if x.session_games else 0, axis=1)
    
    lobby_df['main_roles'] = lobby_df.apply(lambda x: str([n for n in stats_df[stats_df.summoner_name == x.summonerName].team_pos.value_counts()[0:2,].index]), axis=1)


    try:
        lobby_df['on_role'] = lobby_df.apply(lambda x: True if x.main_roles and x.team_pos in x.main_roles else False, axis=1)
    except:
        lobby_df = find_missing_role_and_insert(lobby_df)
        lobby_df['on_role'] = lobby_df.apply(lambda x: True if x.main_roles and x.team_pos in x.main_roles else False, axis=1)

    
    lobby_df['games_on_champ'] = lobby_df.apply(lambda x: len(stats_df[(stats_df.summoner_name == x.summonerName) & (stats_df.champion_name == x.champ_name)]), axis=1)
    lobby_df['wins_on_champ'] = lobby_df.apply(lambda x: len(stats_df[(stats_df.summoner_name == x.summonerName) & (stats_df.champion_name == x.champ_name) & (stats_df.win == True)]), axis=1)
    lobby_df['champ_win_rate'] = lobby_df.apply(lambda x: x.wins_on_champ / x.games_on_champ if x.games_on_champ else 0, axis=1)
    lobby_df['games_on_role'] = lobby_df.apply(lambda x: len(stats_df[(stats_df.summoner_name == x.summonerName)&(stats_df.team_pos == x.team_pos)]),axis=1)
    lobby_df['wins_on_role'] = lobby_df.apply(lambda x: len(stats_df[(stats_df.summoner_name == x.summonerName)&(stats_df.team_pos == x.team_pos) & (stats_df.win == True)]), axis=1)
    lobby_df['role_win_rate'] = lobby_df.apply(lambda x: x.wins_on_role / x.games_on_role if x.games_on_role else 0,axis=1)
    lobby_df['on_role'] = lobby_df.on_role.apply(lambda x: int(x))
    
    lobby_df['blue_jg_on_role'] = lobby_df.apply(lambda x: 1 if x.is_blue_team and x.team_pos == 'JUNGLE' and x.on_role else 0,  axis=1)
    lobby_df['red_jg_on_role'] = lobby_df.apply(lambda x: 1 if not x.is_blue_team and x.team_pos== 'JUNGLE' and x.on_role else 0, axis=1)
    
    return lobby_df


def combine_team_aggregates(blue_team, red_team):
    
    total_aggregate = pd.DataFrame()
        
    for col in blue_team.columns:
        if col not in red_team.columns:
            red_team[col] = 0
            
        total_aggregate[col] = blue_team[col] - red_team[col]
        
    return total_aggregate


def aggregate_lobby(finished_lobby_df):
    total_aggregate = pd.DataFrame()
    blue_aggregate = pd.DataFrame()
    red_aggregate = pd.DataFrame()
    
    
    game_id = finished_lobby_df.match_id.values[0]
    
    blue_team = finished_lobby_df[finished_lobby_df.is_blue_team==1]
    red_team = finished_lobby_df[finished_lobby_df.is_blue_team ==0]
    
    blue_aggregate['total_on_role'] = [blue_team.on_role.sum()]
    blue_aggregate['is_blue_team'] = [blue_team.is_blue_team.mean()]
    
    if blue_team.games_on_champ.sum():
        blue_aggregate['mean_champ_win_rate'] = [blue_team.wins_on_champ.sum() / blue_team.games_on_champ.sum()]
    
    if blue_team.games_on_role.sum():
        blue_aggregate['mean_role_win_rate']  = [blue_team.role_win_rate.mean()]
    
    blue_aggregate['jg_on_role'] = [blue_team.blue_jg_on_role.sum()]
    
    if blue_team.session_games.sum():
        blue_aggregate['mean_session_games'] = [blue_team.session_games.mean()]
        blue_aggregate['aggregate_session_win_rate'] = [blue_team.session_wins.sum() / blue_team.session_games.sum()]
    
    blue_aggregate['win'] = blue_team.win.mean()
    
    red_aggregate['total_on_role'] = [red_team.on_role.sum()]
    
    if red_team.games_on_champ.sum():
        red_aggregate['mean_champ_win_rate'] = [red_team.wins_on_champ.sum() / red_team.games_on_champ.sum()]
    
    if red_team.games_on_role.sum():
        red_aggregate['mean_role_win_rate']  = [red_team.wins_on_role.sum() / red_team.games_on_role.sum()]
    
    red_aggregate['jg_on_role'] = [red_team.red_jg_on_role.sum()]
    red_aggregate['jg_adv'] = [red_team.red_jg_on_role.sum() - blue_team.blue_jg_on_role.sum()]    
    blue_aggregate['jg_adv'] = [blue_team.blue_jg_on_role.sum() - red_team.red_jg_on_role.sum()]    

    red_aggregate['is_blue_team'] = [red_team.is_blue_team.mean()]
    
    if red_team.session_games.sum():
        red_aggregate['mean_session_games'] = [red_team.session_games.mean()]
        red_aggregate['aggregate_session_win_rate'] = [red_team.session_wins.sum() / red_team.session_games.sum()]
    
    red_aggregate['win'] = red_team.win.mean()
    
    
    total_aggregate = combine_team_aggregates(blue_aggregate, red_aggregate)
    total_aggregate['match_id'] = [game_id]
    
    return total_aggregate, blue_aggregate, red_aggregate


def aggregate_all_stats():
    folder = '/Users/kylegreen/codeup-data-science/time-series-exercises/riot_api_data/'
    _, _, filenames = next(os.walk(folder), (None, None, []))
    
    final_df = pd.DataFrame()
    all_red = pd.DataFrame()
    all_blue = pd.DataFrame()
    
    for file in filenames:
        if 'lobby' in file:
            lobby_df = create_lobby_features(folder, file)
            #print(lobby_df)
            agg_lobby_df, blue_df, red_df = aggregate_lobby(lobby_df)
            
            all_red = pd.concat([all_red, red_df])
            all_blue = pd.concat([all_blue, blue_df])
            final_df = pd.concat([final_df, agg_lobby_df])
            
    final_df = final_df.drop_duplicates().reset_index()
    all_red = all_red.drop_duplicates().reset_index()
    all_blue = all_blue.drop_duplicates().reset_index()
    
    all_team_data = pd.concat([all_red, all_blue])
    all_team_data = all_team_data.drop_duplicates()

    return final_df, all_blue, all_red, all_team_data

def aggregate_and_model():

    folder = '/Users/kylegreen/codeup-data-science/time-series-exercises/riot_api_data/'

    aggregates = aggregate_all_stats()
    aggregates.to_csv(folder+'aggregates.csv', index=False)
    aggregates = aggregates.drop(columns=['index'])
    aggregates.dropna(inplace=True)
    print(aggregates)
    print(aggregates.size)

    train, validate, test = split.train_validate_test_split(aggregates, 'win')

    dt_mods, rf_mods, knn_mods, lr_mods= md.all_reports(train, validate, test, 'win')
    return md.Results.total_summary[['model_type', 'train_accuracy', 'validate_accuracy']].sort_values(by='validate_accuracy')

def lobby_and_player_stats(match_id):
    lobby = f'riot_api_data/{match_id}_lobby.csv'
    histories = f'riot_api_data/{match_id}_histories.csv'

    match = api_exp.main(match_id)
    df = match.player_stats
    #df.last_100_games = df.last_100_games.astype('object')
    df['is_session'] = df.time_since_last_game /60 < 46
    
    df.to_csv(lobby, index=False) 
    df = pd.read_csv(lobby)
    #print(df)
    time.sleep(1)
    
    if os.path.isfile(histories):
        player_stats = pd.read_csv(histories)
    else:
        player_stats = pd.DataFrame()
        
    player_stats = get_stat_histories(player_stats, match, filename=histories)
        
    return df, player_stats

def get_stat_histories(df, match_obj, filename, start_id = "", game_info=[], stopped=0, start=0):
    first_loop = True
    ids = list(match_obj.game_history_dict.keys())
    #print(ids)
    
    
    cum_sum = np.cumsum([len(val) for val in match_obj.game_history_dict.values()])
    games = [val for val in match_obj.game_history_dict.values()]
    for i, num in enumerate(cum_sum):
        #print(df.shape[0]-num)
        if df.shape[0]-num < 0:
            index_num = i
            id_start = ids.index(ids[i])
            start = df.shape[0]-num
            #print(id_start)
            break
        else:
            id_start = 0
    
    if start_id:
        id_start = ids.index(start_id)
        #print(id_start)

    for puuid in ids[id_start:]:
        #print(ids[id_start])
        if first_loop:
            counter = start
        else:
            counter = 0
        #print(counter)
        for i, game in enumerate(match_obj.game_history_dict[puuid][counter:]):
            
            if game != '0' and game != 0:
                r = requests.get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{game}', headers=env.headers)
                response = r.json()
                time.sleep(1.225)
                game_duration = response['info']['gameDuration']
                if game_duration /10_000 > 1:
                    game_duration /= 1_000

                try:
                    for participant in response['info']['participants']:
                        #print(response['info']['participants'])

                        if participant['puuid'] == puuid:  
                            try:
                                game_info.append({
                                    'puuid': puuid,
                                    'summoner_name': match_obj.puuid_to_summoner[puuid],
                                    'game_id': game,
                                    'champion_id': participant['championId'],
                                    'champion_name': participant['championName'],
                                    'team_pos': participant['teamPosition'],
                                    'win': participant['win'],
                                    'game_start': response['info']['gameStartTimestamp'],
                                    'game_end': response['info']['gameStartTimestamp']+game_duration,
                                    'game_duration': game_duration,
                                    'team': participant['teamId']
                                })
                            except:
                                try:
                                    game_info.append({
                                        'puuid': puuid,
                                        'summoner_name': match_obj.puuid_to_summoner[puuid],
                                        'game_id': game,
                                        'champion_id': participant['championId'],
                                        'champion_name': participant['championName'],
                                        'team_pos': participant['teamPosition'],
                                        'win': participant['win'],
                                        'game_start': response['info']['gameStartTimestamp'],
                                        'game_end': response['info']['gameEndedInSurrender'],
                                        'game_duration': game_duration,
                                        'team': participant['teamId']
                                    })
                                except:
                                    game_info.append({
                                        'puuid': puuid,
                                        'summoner_name': match_obj.puuid_to_summoner[puuid],
                                        'game_id': game,
                                        'champion_id': participant['championId'],
                                        'champion_name': participant['championName'],
                                        'team_pos': participant['teamPosition'],
                                        'win': participant['win'],
                                        'game_start': response['info']['gameStartTimestamp'],
                                        'game_end': response['info']['gameStartTimestamp'] + game_duration, 
                                        'game_duration': game_duration,
                                        'team': participant['teamId']
                                    })

                            first_loop = False
                            df = pd.concat([df, pd.DataFrame(game_info)])
                            df = df.drop_duplicates()
                            df.to_csv(filename, index=False)

                    counter+= 1
                    print(f'\r{counter} of {len(match_obj.game_history_dict[puuid])} complete\
                    current_id :{puuid}, response: {r.status_code}', end='')

                except KeyError:
                    print(response)
                    stopped+=1
                    if stopped==2:
                        print('stopped')
                        df = pd.concat([df, pd.DataFrame(game_info)])
                        df = df.drop_duplicates()
                        df.to_csv(filename, index=False)
                        return df
                    print(f'we stopped at {counter}')
                    time.sleep(120)
                    if counter == len(game_info):
                        print(f'We can start over at index {counter}')
                    if counter == len(game_info) -1:
                        print(f'Start over at {counter}')
                    df = pd.concat([df, pd.DataFrame(game_info)])
                    df = df.drop_duplicates()
                    df.to_csv(filename, index=False)
                    
                    get_stat_histories(df, match_obj, filename, start_id=puuid, game_info=game_info, stopped=stopped, start=i)
                
                except KeyboardInterrupt:
                    return df
                

    return pd.DataFrame(game_info)