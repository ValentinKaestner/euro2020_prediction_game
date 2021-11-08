
from datetime import date
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
from itertools import product

class Team:
    """
    A class to represent a football team.

    ...

    Attributes
    ----------
    name (str): Name of the team
    elo (str): Elo rating of the team

    Methods
    -------
    expected_outcome(self, other, neutral, homeadv, divisor, base):
        Calculates the expected outcome (W_e) of a football match between two teams based on the Elo rating difference.

    update_elo(self, other, goals_self, goals_other, neutral, homeadv, divisor, base, K, K_factor):
        Updates the Elo ratings of two teams based on the true match result.
    """

    def __init__(self, name):
        self.name = name
        self.elo = 1500  # Set the initial Elo value of 1500

    def expected_outcome(self, other, neutral, homeadv, divisor, base):
        """Calculates the expected outcome W_e between the two national teams `self` and `other` based on the formula
        1 / (1 + base ** ((other.elo - (self.elo + homeadv)) / divisor)).
        ...

        Args:
        -----
            self (Team): The home team.
            other (Team): The away team.
            neutral (bool): True if the match takes place in a neutral location (no home advantage).
            homeadv (float): The additional boost on the Elo due to the home advantage.
            divisor (float): Divisor in the W_e calculation
            base (float): Base in the W_e calculation

        Returns:
        --------
            W_e (float): The expected outcome as a value between 0 (i.e. the home team loses with certainty) and 1 (the home team wins with certainty).

        """
        if neutral == True:  # Is there a home advantage to consider?
            return 1 / (1 + base ** ((other.elo - self.elo) / divisor))
        else:
            return 1 / (1 + base ** ((other.elo - (self.elo + homeadv)) / divisor))

    def update_elo(
        self,
        other,
        goals_self,
        goals_other,
        neutral,
        homeadv,
        divisor,
        base,
        K,
        K_factor,
    ):
        """Updates the Elo ratings of two teams based on the true match result.

        ...

        Args:
        -----
            self (Team): Team A
            other (Team): Team B
            goals_self (int): Goals scored by Team A in the match between Team A and Team B
            goals_other (int): Goals scored by Team B in the match between Team A and Team B
            neutral (bool):
            homeadv (float): The additional boost of a home advantage on the ELO.
            divisor (float):
            base (float):
            K (float):
            K_factor(float):

        Returns:
        --------
            None.

        """
        # Calculate the actual outcome W
        goal_diff = goals_self - goals_other
        if goal_diff > 0:
            W = 1
        elif goal_diff == 0:
            W = 0.5
        else:
            W = 0

        # Calculate the expected outcome W_e
        W_e = self.expected_outcome(other, neutral, homeadv, divisor, base)

        # Calculate the factor to multiply with K based on the goal difference
        K_factor = abs(goal_diff) * K_factor

        # Calculate K_final
        K_final = K * (1 + K_factor)

        # Update both Elo ratings
        self.elo = self.elo + K_final * (W - W_e)
        other.elo = other.elo + K_final * (W_e - W)

def annotate_plot(ax, ys, xs=None, label_format="{:.2f}", fontsize=12, color="grey"):
    """Annotate some plot on axis `ax` with values `ys`.
    
    ...

    Args:
    -----
        ax (): Axis that shall be annotated.
        ys (iterable): y values to be annotated
        xs (iterable): x values. If omitted, then xs is set to range(0, len(ys)).
        label_format (str): The label format of the annotations (Default: "{:.2f}"). 
        fontsize (int): The font size of the annotations (Default: 12).
        color (str): Font color (Default: "grey").

    Returns:
    --------
        None

    """
    if not xs:
        xs = range(0, len(ys))

    for x, y in zip(xs, ys):
        label = label_format.format(y)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color=color,
            weight="bold",
            fontsize=fontsize,
        )

def declutter_plot(ax, directions=["left", "right", "top", "bottom"]):
    """Declutters some plot on axis `ax` by removing the axis labels and yticks.
    ...

    Args:
    -----
        ax (): Axis to declutter.
        directions (list):  

    Returns:
    --------
        None
    
    Raises:
    -------
        AssertionError: Argument `directions` is not a subset of ['left', 'right', 'top', 'bottom']
    
    """
    assert set(directions) <= set(["left", "right", "top", "bottom"]), "Argument `directions` is not a subset of ['left', 'right', 'top', 'bottom']"

    for direction in directions:
        ax.spines[direction].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])

def elo_prob(HELO:float, AELO:float, homeadv:int=0, base:int=7, divisor:int=250):
    '''Returns the winning probability of the home team with a rating of HELO against the away team AELO
    
    Args:
        HELO (float): ELO rating of the home team.
        AELO (float): ELO rating of the away team.
        homeadv (bool): Home advantage to add to the home team's rating.
        base (int): Base parameter of the expected winning probability formula (Default: 7).
        divisor (int): Divisor parameter of the expected winning probability formula (Default: 250).
        
    
    Returns:
        Winning probability of the home team.
    '''
    return 1 / (1 + base ** ((AELO - (HELO + homeadv)) / divisor))

def expected_kt_points(df):
    """Returns a pandas Series with the expected Kicktipp points for some dataframe df with the probabilities
    
    Args:
        df (pandas DataFrame): DataFrame containing probabilities for each result
    
    Returns:
        pandas Series
    """
    EP_list = []
    for i, row in df.iterrows():
        HTipp = i[0]
        ATipp = i[1]
        EP = 0 # Expected Points
        for j, row2 in df.iterrows():
            HTrue = j[0]
            ATrue = j[1]
            EP += kicktipp_points(HTipp, ATipp, HTrue, ATrue) * row2['Prob']
        EP_list.append(EP)
    return pd.Series(EP_list, name="EP")

def get_rating_own_elo(team, elos):
    '''Returns ELO rating of some team at due date'''
    return elos.loc[elos['Team'] == team, 'Elo'].iloc[0]

def identify_rel_matches(team:str, matches:pd.DataFrame, ELO:float, elos:pd.DataFrame, ELO_range=200):
    """Derive column 'Similar Opponent' that flags if opponent is situated in a certain Elo range
    
        Args:
            team (str): The relevant team name.
            matches (pd.DataFrame): DataFrame containing the matches.
            ELO (float): Elo value around which the Elo range is built.
            elos (pd.DataFrame): DataFrame containing the current Elo ratings.
            ELO_range (int): The parameter to define the widthness of the Elo range.
    
        Returns:
            Number of points awarded by kicktipp
    """
    matches['Opponent'] = matches.apply(lambda x: x.away_team if x.home_team == team else x.home_team, axis=1)
    matches['ELO Opponent'] = matches.apply(lambda x: get_rating_own_elo(x['Opponent'], elos), axis=1)
    matches['Similar Opponent'] = (matches['ELO Opponent'] < ELO + ELO_range) & (matches['ELO Opponent'] > ELO - ELO_range)       
    return matches

def kicktipp_points(HTipp, ATipp, HTrue, ATrue):
    '''Returns the number of points for guessing HTipp:ATipp if HTrue:ATrue is the true outcome
    
    Args:
        HTipp (int): Prediction of goals scored by home team.
        ATipp (int): Prediction of goals scored by away team.
        HTrue (int): Actual goals scored by home team.
        ATrue (int): Actual goals scored by away team.
    
    Returns:
        Number of points awarded by kicktipp
    '''
    # Correct result: 4 points
    if (HTipp == HTrue) & (ATipp == ATrue): 
        return 4
    # Correct winner and correct goal difference: 3 points
    elif (HTipp - ATipp == HTrue - ATrue) & (HTipp != ATipp): # 
        return 3
    # Correct winner with wrong goal difference or incorrect draw: 2 points
    elif ((HTipp == ATipp) & (HTrue == ATrue)) | ((HTipp > ATipp) & (HTrue > ATrue)) | ((HTipp < ATipp) & (HTrue < ATrue)):
        return 2
    # else 0
    else:
        return 0

def kt_points(diff_predict, diff_true):
    """Returns the number of points in Kicktipp for predicting a certain goal difference given the true goal difference.
    Number of points is derived from the assumption that the correct final score is predicted in 50% of the cases (if there is a winner)
    or 40% (draw).
    
    ...
    
    Args:
    -----
        diff_predict (int): The predicted goal difference in a match between Team A and Team B (Goals Team A - Goals Team B).
        diff_true (int): The true goal difference in a match between Team A and Team B (Goals Team A - Goals Team B).
    
    Returns:
    --------
        Number of Kicktipp points
    """
    if (diff_predict == diff_true) & (diff_predict != 0):
        return 3.5 
    elif ((diff_predict == diff_true) & (diff_predict == 0)):
        return 2.8 
    elif (diff_predict * diff_true > 0):
        return 2
    else:
        return 0

def last_n_matches_team(team:str, n_games:int, due_date=str(date.today())):
    '''Returns a dataframe from the last n games involving a specific team
    
    Args:
        team (str): Name of the team
        n_games (int): Number of matches to return
        due_date (str): String containing the due date. Only matches before the due date are considered. Default is today.
    
    Returns:
        Pandas DataFrame containing the last n matches involving `team`
    '''
    df_matches = pd.read_csv('files/results.csv', parse_dates=['date'])
    df_matches = df_matches[df_matches["date"] < due_date]
    df_matches_team = df_matches[(df_matches['home_team'] == team) | (df_matches['away_team'] == team)]
    df_matches_team = df_matches_team.sort_values('date', ascending=False)
    return df_matches_team.head(n_games)

def mean_goals(df:pd.DataFrame, team:str, n_games:int=10):
    '''Returns the mean number of goals scored in the last n games involving a specific team
    
    Args:
        df (pd.DataFrame): DataFrame containing two columns `home_score` and `away_score`.
        team (str): Name of the relevant team.
        n_games (int): Number of games on which the mean shall be calculated.
    
    Returns:
        Float
    '''
    if isinstance(df, pd.DataFrame) == False:
        df = last_n_matches_team(team, n_games)
    sum_goals = df['home_score'].sum() + df['away_score'].sum()
    return sum_goals / df.shape[0]

def param_grid_search_generator(**param_iterators):   
    """Copied from https://stackoverflow.com/questions/42139511/iterator-for-looping-over-ranges-of-parameters-in-subsets-of-dictionary-keys"""
    param_names = list(param_iterators.keys())
    param_combination_generator = product(*list(param_iterators.values()))
    for param_combination in param_combination_generator:
        yield {param_names[i]: param_combination[i] for i in range(len(param_names))}

def result(team_home, team_away, elos, home_advantage=False, n_output_rows=5, n_games=10, sort_col='EP', ELO_range=150, due_date=str(date.today())):
    """Prints relevant match information and returns result that maximises the Kicktipp result for a match between team_home and team_away
    
    ...

    Args:
    -----
        team_home (str): The home team. 
        team_away (str): The away team.
        elos (pd.DataFrame): DataFrame containing the current Elo ratings. 
        home_advantage (int): Home advantage boost to the rating (Default: False)
        n_output_rows (int): Number of rows to output (Default: 5)
        n_games (int): Number of last games to output per team (Default: 10)
        sort_col (str): Column to sort by (Default: "EP")
        ELO_range (int): Parameter to identify opponents with similar rating as the current opponent (Default: 150)
        due_date (str): The date of the match (Default: Today).

    Returns:
    --------
        Formatted DataFrame with result predictions

    """
    
    # Retrieve current ELO ratings of both teams
    ELO_home = get_rating_own_elo(team_home, elos)
    ELO_away = get_rating_own_elo(team_away, elos)
    print('ELO', team_home, ':', format(ELO_home, ".0f"))
    print('ELO', team_away, ':', format(ELO_away, ".0f"))
    print('\n')
    
    # Calculate estimate for the mean number of goals for each team
    # --> Maximum Likelihood Estimate for the Poisson parameter lambda
    team_home_last_matches = last_n_matches_team(team_home, n_games, due_date)
    team_away_last_matches = last_n_matches_team(team_away, n_games, due_date)

    
    team_home_last_matches_rel = identify_rel_matches(team_home, team_home_last_matches, ELO_away, elos, ELO_range)
    team_away_last_matches_rel = identify_rel_matches(team_away, team_away_last_matches, ELO_home, elos, ELO_range)

    ELO_range_home = ELO_range
    ELO_range_away = ELO_range

    while team_home_last_matches_rel["Similar Opponent"].sum() < 3:
        ELO_range_home = ELO_range_home + 20
        team_home_last_matches_rel = identify_rel_matches(team_home, team_home_last_matches, ELO_away, elos, ELO_range_home)

    while team_away_last_matches_rel["Similar Opponent"].sum() < 3:
        ELO_range_away = ELO_range_away + 20
        team_away_last_matches_rel = identify_rel_matches(team_away, team_away_last_matches, ELO_home, elos, ELO_range_away)

    print('Last', n_games, 'matches of', team_home, ':')
    print(team_home_last_matches_rel[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'Similar Opponent']])
    print('\n')
    print('Last', n_games, 'matches of', team_away, ':')
    print(team_away_last_matches_rel[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'Similar Opponent']])
    print('\n')
    
    # Select only matches where opponent had a similar ELO rating (i.e. opponent ELO +-300)
    team_home_rel_matches = team_home_last_matches_rel[team_home_last_matches_rel['Similar Opponent'] == True]
    team_away_rel_matches = team_away_last_matches_rel[team_away_last_matches_rel['Similar Opponent'] == True]
    
    lambda_home = mean_goals(df=team_home_rel_matches)
    lambda_away = mean_goals(df=team_away_rel_matches)
    lambda_total = (lambda_home + lambda_away) / 2
    print('Mean number of goals with', team_home, 'involved:', lambda_home)
    print('Mean number of goals with', team_away, 'involved:', lambda_away)
    print('\n')
    
    # Calculate winning probability of home team
    win_prob = elo_prob(ELO_home, ELO_away, home_advantage)
    
    # Incorporate the winning probability into the Poisson parameters for both teams
    lambda_home_elo = lambda_total * win_prob
    lambda_away_elo = lambda_total * (1-win_prob)
    
    # Simulate results and find the probabilities for each result
    df_results = result_probs(lambda_home_elo, lambda_away_elo)
    
    # Calculate expected Kicktipp points for each simulated result
    result_expected_points = expected_kt_points(df_results).to_frame()
    result_expected_points.index = df_results.index
    
    # Add expected Kicktipp point as separate column in df_results
    df_results['EP'] = result_expected_points
    df_results.index.set_names([team_home, team_away], inplace=True)
    
    return df_results.sort_values(sort_col, ascending=False).head(n_output_rows).style.format({'Prob':'{:.2%}', 'EP': '{:.3f}'})

def result_probs(lambda_home, lambda_away, n_sims:int=1000000):
    '''Simulates `n_sims` observations of two Poisson random variables (one for home goals, one for away goals)
        and returns a dataframe with the relative frequencies of each simulated result.
        
    Args:
        lambda_home (float): Poisson lambda parameter of the home team.
        lambda_away (float): Poisson lambda parameter of the away team.
        n_sims (int): Number of observations to simulate (Default: 1 million).
        
    Returns:
        pandas DataFrame
    '''
    goals_home = np.random.poisson(lambda_home, n_sims)
    goals_away = np.random.poisson(lambda_away, n_sims)
    df = pd.DataFrame(np.hstack((goals_home[:,None], goals_away[:,None])), columns=["Goals Home", "Goals Away"])
    df_counts = df.value_counts(subset=["Goals Home", "Goals Away"], normalize=True)
    return df_counts.to_frame(name="Prob")

def simulate_elos(df, homeadv:int, divisor:int, base:int, K:float, K_factor:float, due_date=str(date.today()), plot_linreg=False):
    """Simulates historical Elo ratings

    ...

    Args:
    -----
        df (pd.DataFrame): Dataframe with the match results
        homeadv (int): The additional boost of a home advantage on the Elo rating.
        divisor (int): The divisor parameter in the Elo rating calculation.
        base (int): The base parameter in the Elo rating calculation.
        K (float): The development factor in the Elo rating calculation.
        K_factor (float): The goal difference driven multiplier in the Elo rating calculation.
        due_date (str): Only matches before the due_date are considered (Default: today).
        plot_linreg (Boolean): True if the fitted linear regression shall be plotted (Default: False)

    Returns:
    --------
        avg_points (float): The average amount of Kicktipp points achieved by predicting the goal difference.
        ratings (DataFrame): The final Elo ratings.
        stdev (float): Standard deviation of Kicktipp points.
        df_2000 (DataFrame): DataFrame containing the match results starting from 2000 as well as the predictions.
    """

    # Initialisation
    Home_ELO_New = []
    Home_ELO_Old = []
    Away_ELO_New = []
    Away_ELO_Old = []
    team_dict = {}

    # Loop through all matches in df
    for index, row in df.iterrows():
        # Create new instance of class Team at the first occurrence
        for team in [row["home_team"], row["away_team"]]:
            # team_dict.setdefault(team, Team(team))
            if team not in team_dict:
                team_dict[team] = Team(team)
        # Fetch and store current Elo ratings (before the game)
        Home_ELO_Old.append(team_dict[row["home_team"]].elo)
        Away_ELO_Old.append(team_dict[row["away_team"]].elo)
        # Update Elo ratings based on the actual game result
        team_dict[row["home_team"]].update_elo(
            team_dict[row["away_team"]],
            row["home_score"],
            row["away_score"],
            row["neutral"],
            homeadv,
            divisor,
            base,
            K,
            K_factor,
        )
        # Fetch and store updated Elo ratings in lists
        Home_ELO_New.append(team_dict[row["home_team"]].elo)
        Away_ELO_New.append(team_dict[row["away_team"]].elo)

    # Save the ELO ratings history in separate DataFrame columns
    df["home_team_elo_old"] = pd.Series(Home_ELO_Old)
    df["away_team_elo_old"] = pd.Series(Away_ELO_Old)
    df["home_team_elo_new"] = pd.Series(Home_ELO_New)
    df["away_team_elo_new"] = pd.Series(Away_ELO_New)

    # Set up a validation set (UEFA Euro and FIFA World Cup matches starting in 2000)
    df_2000 = df[
        (df["date"] > "1999-12-31")
        & (df["date"] < due_date)
        & ((df["tournament"] == "UEFA Euro") | (df["tournament"] == "FIFA World Cup"))
    ].copy()
    df_2000["diff_score"] = df["home_score"] - df["away_score"]
    df_2000["diff_elo"] = df["home_team_elo_old"] - df["away_team_elo_old"]

    # Fit a linear regression model (goal_difference = b + m * elo_difference)
    b, m = polyfit(df_2000["diff_elo"], df_2000["diff_score"], 1)
    df_2000["exp_goal_diff_lin_reg"] = b + m * df_2000["diff_elo"]
    
    # Plot the linear regression fit
    if plot_linreg == True:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        plt.scatter(df_2000["diff_elo"], df_2000["diff_score"])
        plt.plot(range(-600, 600, 10), [b + m * x for x in range(-600, 600, 10)], )
        
        # Add title and labels
        plt.title("The Relationship between Rating and Goal Differences")
        plt.xlabel("Elo Rating Difference")
        plt.ylabel("Goals Difference")
        
        plt.show()
        return None

    # Use the linear regression model to predict the goal difference
    df_2000["kt_diff"] = round(df_2000["exp_goal_diff_lin_reg"], 0)
    # If the expected goal difference is between -0.5 and 0.5, then round to -1 and 1 (no draw predictions)
    df_2000.loc[(df_2000["exp_goal_diff_lin_reg"] >= 0) & (df_2000["exp_goal_diff_lin_reg"] <= 0.5), "kt_diff"] = 1
    df_2000.loc[(df_2000["exp_goal_diff_lin_reg"] < 0) & (df_2000["exp_goal_diff_lin_reg"] > -0.5), "kt_diff"] = -1
    
    # Calulate the average Kicktipp points
    n_matches = df_2000.shape[0]
    points = df_2000.apply(lambda x: kt_points(x["kt_diff"], x["diff_score"]), axis=1)
    avg_points = points.sum() / n_matches
    stdev = points.std()

    # Create a DataFrame with the final Elo values for each national team
    ratings = pd.DataFrame(
        {
            "Team": list(team_dict.keys()),
            "Elo": [team.elo for team in team_dict.values()],
        }
    )
    ratings = ratings.sort_values(by="Elo", ascending=False).reset_index(drop=True)

    return (avg_points, ratings, stdev, df_2000)
    











 

