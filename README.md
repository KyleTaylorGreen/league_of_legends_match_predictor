## What:
> * Construct an ML Classifier to predict whether a team in an online league of legends match is going to win or lose.
> * Deliver a final notebook describing the process, insights found, and the classifier itself.

## Why:
> * Ranked Matchmaking is supposed to give both teams an equal footing against each other. A model that could more accurately predict whether a team would win or lose could help identify features to make matches more fair for everyone.

## Hypotheses:
> * Teams with more people on role will perform better on average.
> * Teams with higher mean champion winrate will perform better on average.
> * Teams that have a jungle player on role vs teams without one will perform better on average.


**It's important to note that all numbers are a function of Blue Team's number for each column subtracted by Red Team's number for each column. --> A positive number indicates an advantage for blue team, and a negative number indicates an advantage for red team.

i.e: A value of 1 for 'on_role' indicates that blue team has one more person on their preferred role than red team.

##Data Dictionary:
 * total_on_role: total number of people on their preferred role.
 * mean_role_win_rate: win rate of people on the role they were assigned.
 * mean_champ_win_rate: win rate of people on the champions that they picked for that game.
 * mean_session_games: the amount of games played where less than 45 minutes has elapsed between games.
 * mean_session_win_rate: the win rate of both teams for their sessions
 * jg_on_role: Which team has jg on role. 0 means they both do, 1 means only blue team, -1 means only red team.
 * win: Whether blue team won (1) or red team won (-1)
 * match_id: the id of the match that identifies the aggregate game lobby.

## Executive Summary:

### Key Drivers: 
 * I was unable to establish statistical significance in the features discussed above. 

### Model Performance:
My model had a 68.2% accuracy on test, beating baseline accuracy of 51.7% by a total of 32.5%.

### Takeaways/Recommendations:
* Variability between train (58%), validate (61%) and test(68%) accuracies, leading me to believe that not enough data has been collected. I recommend collecting more data and recalculating the model at a later date.
* Statistical significance was not established with the above features. It could be a lack of data points, or they could really not be indicators of winning/losing teams. 
* More exploration, data acquisition, and potentially feature engineering is necessary before this model is ready to be implemented.