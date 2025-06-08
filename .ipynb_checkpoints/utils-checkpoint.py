import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
import random

# Add the Completed directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Completed'))

def load_fixtures(file_path: str) -> pd.DataFrame:
    """
    Load fixtures from a CSV file.
    
    Args:
        file_path: Path to the fixtures CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing fixtures
    """
    print(f"Loading fixtures from {file_path}...")
    return pd.read_csv(file_path)

def load_team_attributes(file_path: str) -> Dict[float, List[float]]:
    """
    Load team attributes from a CSV file and convert to the format needed by the match engine.
    
    Args:
        file_path: Path to the team attributes CSV file
        
    Returns:
        Dict[float, List[float]]: Dictionary mapping team IDs to attributes [defense, midfield, attack]
    """
    print(f"Loading team attributes from {file_path}...")
    attrs_df = pd.read_csv(file_path)
    
    # Create a dictionary mapping team IDs to attributes [defense, midfield, attack]
    team_attrs = {}
    for _, row in attrs_df.iterrows():
        team_id = row['teamID']
        defense = row['defense']
        midfield = row['midfield']
        attack = row['attack']
        team_attrs[team_id] = [defense, midfield, attack]
    
    return team_attrs

def load_team_names(file_path: str) -> Dict[float, str]:
    """
    Load team names from a CSV file.
    
    Args:
        file_path: Path to the teams CSV file
        
    Returns:
        Dict[float, str]: Dictionary mapping team IDs to names
    """
    print(f"Loading team names from {file_path}...")
    teams_df = pd.read_csv(file_path)
    
    # Create a dictionary mapping team IDs to names
    team_names = {}
    for _, row in teams_df.iterrows():
        team_id = row['teamID']
        team_names[team_id] = row['teamName']
    
    return team_names

def get_cup_rounds(fixtures_df: pd.DataFrame, competition: str) -> List[str]:
    """
    Get all rounds for a cup competition in order.
    
    Args:
        fixtures_df: DataFrame containing fixtures
        competition: Cup competition name
        
    Returns:
        List[str]: List of round names in order
    """
    # Get unique rounds for this competition and sort by round_order
    rounds_df = fixtures_df[fixtures_df['competition'] == competition].drop_duplicates(['round', 'round_order'])
    rounds_df = rounds_df.sort_values('round_order')
    
    return rounds_df['round'].tolist()

def check_for_duplicates(fixtures_df: pd.DataFrame, competition: str) -> bool:
    """
    Check for duplicate team appearances in a cup competition.
    
    Args:
        fixtures_df: DataFrame containing fixtures
        competition: Cup competition name
        
    Returns:
        bool: True if duplicates were found, False otherwise
    """
    cup_fixtures = fixtures_df[fixtures_df['competition'] == competition]
    rounds = cup_fixtures['round'].unique()
    
    found_duplicates = False
    print(f"\nChecking for duplicate team appearances in {competition}...")
    
    for round_name in rounds:
        round_fixtures = cup_fixtures[cup_fixtures['round'] == round_name]
        
        # Get all teams in this round
        teams = []
        for _, match in round_fixtures.iterrows():
            if not pd.isna(match['homeTeam']):
                teams.append(match['homeTeam'])
            if not pd.isna(match['awayTeam']):
                teams.append(match['awayTeam'])
        
        # Check for duplicates
        unique_teams = set(teams)
        if len(unique_teams) < len(teams):
            found_duplicates = True
            print(f"  WARNING: Found duplicate teams in {round_name}!")
            
            # Count occurrences of each team
            team_counts = {}
            for team in teams:
                if team in team_counts:
                    team_counts[team] += 1
                else:
                    team_counts[team] = 1
            
            # Print teams that appear more than once
            for team, count in team_counts.items():
                if count > 1:
                    print(f"    Team {int(team)} appears {count} times")
        else:
            print(f"  {round_name}: No duplicates found ({len(teams)} teams)")
    
    return found_duplicates

def print_league_table(fixtures_df: pd.DataFrame, competition: str, team_names: Dict[float, str]) -> pd.DataFrame:
    """
    Print the league table with team names.
    
    Args:
        fixtures_df: DataFrame containing fixtures
        competition: League competition name
        team_names: Dictionary mapping team IDs to names
        
    Returns:
        pd.DataFrame: DataFrame containing the league table
    """
    # Filter fixtures for the specific competition
    league_fixtures = fixtures_df[(fixtures_df['competition'] == competition) & 
                                 (fixtures_df['homeTeam'].notna()) & 
                                 (fixtures_df['awayTeam'].notna()) &
                                 (fixtures_df['homeGoals'].notna()) & 
                                 (fixtures_df['awayGoals'].notna())]
    
    # Create a dictionary to store team stats
    team_stats = {}
    
    # Process each match
    for _, match in league_fixtures.iterrows():
        home_team = match['homeTeam']
        away_team = match['awayTeam']
        home_goals = match['homeGoals']
        away_goals = match['awayGoals']
        
        # Initialize team stats if not already present
        for team in [home_team, away_team]:
            if team not in team_stats:
                team_stats[team] = {
                    'played': 0,
                    'won': 0,
                    'drawn': 0,
                    'lost': 0,
                    'goals_for': 0,
                    'goals_against': 0,
                    'goal_difference': 0,
                    'points': 0
                }
        
        # Update home team stats
        team_stats[home_team]['played'] += 1
        team_stats[home_team]['goals_for'] += home_goals
        team_stats[home_team]['goals_against'] += away_goals
        
        # Update away team stats
        team_stats[away_team]['played'] += 1
        team_stats[away_team]['goals_for'] += away_goals
        team_stats[away_team]['goals_against'] += home_goals
        
        # Update results and points
        if home_goals > away_goals:
            team_stats[home_team]['won'] += 1
            team_stats[home_team]['points'] += 3
            team_stats[away_team]['lost'] += 1
        elif home_goals < away_goals:
            team_stats[away_team]['won'] += 1
            team_stats[away_team]['points'] += 3
            team_stats[home_team]['lost'] += 1
        else:
            team_stats[home_team]['drawn'] += 1
            team_stats[home_team]['points'] += 1
            team_stats[away_team]['drawn'] += 1
            team_stats[away_team]['points'] += 1
    
    # Calculate goal difference
    for team in team_stats:
        team_stats[team]['goal_difference'] = team_stats[team]['goals_for'] - team_stats[team]['goals_against']
    
    # Create a DataFrame from the team stats
    table_df = pd.DataFrame.from_dict(team_stats, orient='index')
    
    # Sort by points, goal difference, goals for
    table_df = table_df.sort_values(['points', 'goal_difference', 'goals_for'], ascending=[False, False, False])
    
    # Reset index and add position column
    table_df = table_df.reset_index().rename(columns={'index': 'team'})
    table_df.insert(0, 'pos', range(1, len(table_df) + 1))
    
    # Print the table
    print(f"\n{competition} Table:")
    print("=" * 80)
    print(f"{'Pos':>3} {'Team':>25} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GF':>3} {'GA':>3} {'GD':>3} {'Pts':>3}")
    print("-" * 80)
    
    for _, row in table_df.iterrows():
        team_id = row['team']
        team_name = team_names.get(team_id, f"Team {int(team_id)}")
        print(f"{int(row['pos']):3d} {team_name:25s} {int(row['played']):3d} {int(row['won']):3d} {int(row['drawn']):3d} {int(row['lost']):3d} {int(row['goals_for']):3d} {int(row['goals_against']):3d} {int(row['goal_difference']):3d} {int(row['points']):3d}")
    
    return table_df

def track_cup_progression(fixtures_df: pd.DataFrame, cup_name: str, team_names: Dict[float, str]) -> Optional[float]:
    """
    Track the progression of teams through a cup competition.
    
    Args:
        fixtures_df: DataFrame containing fixtures
        cup_name: Cup competition name
        team_names: Dictionary mapping team IDs to names
        
    Returns:
        Optional[float]: The ID of the cup winner, or None if no winner found
    """
    # Get all rounds for this cup
    cup_rounds = get_cup_rounds(fixtures_df, cup_name)
    
    print(f"\n{cup_name} Progression:")
    print("=" * 80)
    print(f"Total rounds: {len(cup_rounds)}")
    print(f"Rounds in order: {', '.join(cup_rounds)}")
    
    # Find the cup final
    cup_final = fixtures_df[(fixtures_df['competition'] == cup_name) & 
                          (fixtures_df['round'] == 'Final')]
    
    if len(cup_final) == 0 or pd.isna(cup_final.iloc[0]['homeGoals']):
        print(f"\nNo {cup_name} winner found - final not played yet")
        return None
    
    cup_final = cup_final.iloc[0]
    
    if cup_final['homeGoals'] > cup_final['awayGoals']:
        winner_id = cup_final['homeTeam']
    else:
        winner_id = cup_final['awayTeam']
    
    if pd.isna(winner_id):
        print(f"\nNo {cup_name} winner found - teams not assigned to final")
        return None
    
    winner_name = team_names.get(winner_id, f"Team {int(winner_id)}")
    
    print(f"\n{cup_name} Winner: {winner_name} (ID: {int(winner_id)})")
    
    # Filter fixtures for the specific cup and where the winner played
    cup_fixtures = fixtures_df[(fixtures_df['competition'] == cup_name) & 
                              ((fixtures_df['homeTeam'] == winner_id) | 
                               (fixtures_df['awayTeam'] == winner_id)) &
                              (fixtures_df['homeGoals'].notna()) &
                              (fixtures_df['awayGoals'].notna())]
    
    # Sort by round_order to show progression
    cup_fixtures = cup_fixtures.sort_values('round_order')
    
    if len(cup_fixtures) == 0:
        print(f"No matches found for {winner_name} in {cup_name}")
        return winner_id
    
    # Display each match
    print(f"\nMatch Progression for {winner_name}:")
    print("-" * 80)
    print(f"{'Round':20s} {'Home Team':25s} {'Score':^7s} {'Away Team':25s}")
    print("-" * 80)
    
    for _, match in cup_fixtures.iterrows():
        home_team_id = match['homeTeam']
        away_team_id = match['awayTeam']
        
        home_name = team_names.get(home_team_id, f"Team {int(home_team_id)}")
        away_name = team_names.get(away_team_id, f"Team {int(away_team_id)}")
        
        score = f"{int(match['homeGoals'])}-{int(match['awayGoals'])}"
        
        # Highlight the winner's name
        if home_team_id == winner_id:
            home_name = f"* {home_name}"
        if away_team_id == winner_id:
            away_name = f"* {away_name}"
        
        print(f"{match['round']:20s} {home_name:25s} {score:^7s} {away_name:25s}")
    
    return winner_id

def count_matches_by_round(fixtures_df: pd.DataFrame, competition: str) -> pd.DataFrame:
    """
    Count the number of matches in each round of a competition.
    
    Args:
        fixtures_df: DataFrame containing fixtures
        competition: Competition name
        
    Returns:
        pd.DataFrame: DataFrame with round counts
    """
    comp_fixtures = fixtures_df[fixtures_df['competition'] == competition]
    
    # Count matches by round
    round_counts = comp_fixtures.groupby('round').size().reset_index(name='total_matches')
    
    # Count matches with results by round
    played_counts = comp_fixtures[comp_fixtures['homeGoals'].notna()].groupby('round').size().reset_index(name='played_matches')
    
    # Merge the counts
    result = pd.merge(round_counts, played_counts, on='round', how='left')
    result['played_matches'] = result['played_matches'].fillna(0).astype(int)
    
    # Add round_order
    round_orders = comp_fixtures.drop_duplicates('round')[['round', 'round_order']]
    result = pd.merge(result, round_orders, on='round', how='left')
    
    # Sort by round_order
    result = result.sort_values('round_order')
    
    return result

def print_match_counts(fixtures_df: pd.DataFrame, competition: str) -> None:
    """
    Print the number of matches in each round of a competition.
    
    Args:
        fixtures_df: DataFrame containing fixtures
        competition: Competition name
    """
    counts = count_matches_by_round(fixtures_df, competition)
    
    print(f"\n{competition} Match Counts:")
    print("=" * 60)
    print(f"{'Round':20s} {'Played':>10s} {'Total':>10s} {'Percentage':>10s}")
    print("-" * 60)
    
    for _, row in counts.iterrows():
        round_name = row['round']
        played = int(row['played_matches'])
        total = int(row['total_matches'])
        percentage = (played / total * 100) if total > 0 else 0
        
        print(f"{round_name:20s} {played:10d} {total:10d} {percentage:10.1f}%")
    
    # Print totals
    total_played = counts['played_matches'].sum()
    total_matches = counts['total_matches'].sum()
    total_percentage = (total_played / total_matches * 100) if total_matches > 0 else 0
    
    print("-" * 60)
    print(f"{'TOTAL':20s} {total_played:10d} {total_matches:10d} {total_percentage:10.1f}%")
