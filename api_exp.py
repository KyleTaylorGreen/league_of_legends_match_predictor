import env
import time
import datetime
import pandas as pd
import cassiopeia


# see https://github.com/psf/requests/issues/1595#issuecomment-504030697 for more details
import requests
import ujson

requests.models.complexjson = ujson



RIOT_API_KEY = env.RIOT_API_KEY
my_puuid = 'YUYpGhggndqqGjTbJf_vfCO8r6pEwepm8PTLobOkX9dFVKV5B5QDGhrElvukJfSwSCw4d_4ipPKpFA'
base_url = 'https://americas.api.riotgames.com'
# solo ranked summoner's rift 5v5
queue_id = 420
settings= {"RiotAPI": {
                      "api_key": "RIOT_API_KEY",
                      "limiting_share": 1.0,
                      "request_error_handling": {
        "404": {
            "strategy": "throw"
        },
        "429": {
            "service": {
                "strategy": "exponential_backoff",
                "initial_backoff": 1.0,
                "backoff_factor": 2.0,
                "max_attempts": 4
            },
            "method": {
                "strategy": "retry_from_headers",
                "max_attempts": 5
            },
            "application": {
                "strategy": "retry_from_headers",
                "max_attempts": 5
            }
      },
      "500": {
          "strategy": "throw"
      },
      "503": {
          "strategy": "throw"
      },
      "timeout": {
          "strategy": "throw"
      }
    }
}
}
cassiopeia.apply_settings(settings)
urls= []

def get_match_history(puuid):
    start=  0
    count= 100

    response = requests.get(f'{base_url}/lol/match/v5/matches/by-puuid/{puuid}/ids?queue={queue_id}&start={start}&count={count}', headers=env.headers)
    return response.json()

def get_match_data_from(match):
    """returns dictionary of match id as key and 
    match data as value"""

    match_data = {}
    
    match_stats_url = f'/lol/match/v5/matches/{match}'
    response = requests.get(f'{base_url}{match_stats_url}', headers=env.headers).json()
    
    match_data[match] = response

    return match_data


class Match:
    all_matches = {}

    def __init__(self,matchid):
        self.player_stats = pd.DataFrame()
        
        self.matchid = matchid
        self.match_data = get_match_data_from(matchid)
        self.participants = self.get_participants()
        self.start_u, self.game_duration, self.creation = self.get_game_time()

        self.puuids = self.match_data[self.matchid]['metadata']['participants']

        self.start_dt = datetime.datetime.fromtimestamp(self.start_u/1000)

        self.blue_team, self.red_team = self.get_teams()
        self.blue_won = self.get_team_win()
        self.red_won = not self.blue_won

        self.players = {}
        self.game_history_dict = {}
        self.puuid_to_summoner = {}
        for puuid in self.puuids:
            time.sleep(1)
            summoner = requests.get(f'https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}', headers=env.headers).json()
            if PlayerAtTimeOfGame(puuid, self).games_before_match:
                previous_game_id = PlayerAtTimeOfGame(puuid, self).games_before_match
                print(previous_game_id)
            else:
                previous_game_id = [0]

            if previous_game_id[0]:
                last_game_data = get_match_data_from(previous_game_id[0])
                try:
                    if last_game_data[previous_game_id[0]]['info']['gameDuration'] / 10_000 > 1:
                        duration = last_game_data[previous_game_id[0]]['info']['gameDuration'] / 1000
                    else:
                        duration = last_game_data[previous_game_id[0]]['info']['gameDuration']

                    last_game_finish = last_game_data[previous_game_id[0]]['info']['gameStartTimestamp']/1000 + duration
                except: 
                    last_game_finish = 0
                last_100_games = PlayerAtTimeOfGame(puuid, self).get_games_before_current_game()
            else:
                last_game_finish = 0
                last_100_games = [0]

            for participant in self.match_data[self.matchid]['info']['participants']:
                if participant['puuid'] == puuid:
                    win = participant['win']
                    champ = participant['championName']
                    champ_id = participant['championId']
                    team_pos = participant['teamPosition']
            self.game_history_dict[puuid] = last_100_games
            self.puuid_to_summoner[puuid] = summoner['name']
            self.players[puuid] = {'puu_id': puuid,
                                   'match_id': self.matchid,
                                   'previous_game_id': previous_game_id[0],
                                   'account_id' : summoner['accountId'],
                                   'summonerName': summoner['name'],
                                   'id': summoner['id'],
                                   'game_creation_time': self.creation,
                                   'game_start': self.start_u,
                                   'game_finish': self.start_u + (self.game_duration * 1000),
                                   'game_duration': self.game_duration,
                                   'win': win,
                                   'champ_name': champ,
                                   'champ_id': champ_id,
                                   'team_pos': team_pos,
                                   'time_since_last_game': (self.start_u - last_game_finish)/60,
                                   'last_100_games': last_100_games
                                   }
        
            
        self.player_stats = pd.DataFrame(self.players.values())
        self.all_matches[self.matchid] = self.player_stats
            
    def get_participants(self):
        #print(self.match_data)
        participants = []

        for participant in self.match_data[self.matchid]['info']['participants']:
            participants.append(participant['summonerName'])
        
        return participants

    def get_teams(self):
        blue_team = []
        red_team = []

        for team in self.match_data[self.matchid]['info']['participants']:
            if team['teamId'] == 100:
                blue_team.append(team['summonerName'])
            else:
                red_team.append(team['summonerName'])

        return blue_team, red_team
    

    def get_game_time(self):
        start = self.match_data[self.matchid]['info']['gameStartTimestamp']/1000
        try:
            duration = self.match_data[self.matchid]['info']['gameDuration']
        except:
            duration = 0
        if duration / 10_000 > 1:
            duration/=1000
        creation = self.match_data[self.matchid]['info']['gameCreation'] /1000

        return start, duration, creation

    def get_team_win(self):
        for team in self.match_data[self.matchid]['info']['participants']:
            if team['teamId'] == 100:
                if team['win'] == True:
                    return True
                else:
                    return False

class PlayerAtTimeOfGame:

    def __init__(self, puuid, match):
        self.puuid = puuid
        self.match_data = match.match_data
        self.matchid = match.matchid
        self.creation = self.match_data[match.matchid]['info']['gameCreation']
        self.games_before_match = self.most_recent_game()

    def get_games_before_current_game(self, count=100):
        limiting_time = int(self.creation/1000) - 1
        print(f'CREATION TIME!!!: {self.creation/1000}')
        response = requests.get(f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?endTime={limiting_time}&queue=420&start=0&count={count}', headers=env.headers).json()

        return response

    def most_recent_game(self, count=1):
        return self.get_games_before_current_game(count=count)

def main(basic_match_id):
    match_info = Match(basic_match_id)

    return match_info

#if __name__ == '__main__':
    basic_match_id = 'NA1_4169229565'
    NA1_4169229565 = Match(basic_match_id)
    
    
    
#class Player:
#    def __init__(self, ):
