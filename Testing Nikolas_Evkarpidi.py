#!/usr/bin/env python
# coding: utf-8
import csv

import numpy as np
import pandas as pd
import scipy.stats

lines = []
# Reading files and fixing the bugs in collected data
with open('C:/Users/79181/PycharmProjects/pythonProject/user.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    for row in reader:
        # all rows with len != 3 are wrongly inputed
        if len(row) == 3:
            lines.append(row)

with open('C:/Users/79181/PycharmProjects/pythonProject/user_fixed.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)

# Making up the final table with the data
users_df = pd.read_csv("C:/Users/79181/PycharmProjects/pythonProject/user_fixed.csv")
games_df = pd.read_csv("C:/Users/79181/PycharmProjects/pythonProject/games.csv")
games_df = games_df.rename(columns=({'id_game': 'GameID'}))
users_games_df = pd.merge(users_df, games_df,
                          on='GameID',
                          how='inner')
users_games_df['rating'] = pd.to_numeric(users_games_df['rating'], errors='coerce')
users_games_df = users_games_df.dropna()
users_games_df = users_games_df.drop(columns=["genre", "price", "title"])


def calculateRecommendations(inputPreferences):
    # Subset of users with similar games, which will be used to make a suggestion
    userSubset = users_games_df[users_games_df['GameID'].
        isin(inputPreferences[inputPreferences['rating'] > 5]['GameID'].tolist())]
    userSubset = userSubset[userSubset['UserID'] != inputUser]
    userSubsetGroup = userSubset.groupby(['UserID'])
    userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

    # Calculating Pearson Coefficient

    pearsonCorrelationDict = {}

    for name, group in userSubsetGroup:

        # Let’s start by sorting the input and current user group
        group = group.sort_values(by='GameID')
        inputPreferences = inputPreferences.sort_values(by='GameID')

        # Get the N for the formula
        nRatings = len(group)
        # Get the review scores for the games that they both have in common
        temp_df = inputPreferences[inputPreferences['GameID'].isin(group['GameID'].tolist())]
        # And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp_df['rating'].tolist()
        # Let’s also put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()

        no_constants = (np.asarray(tempRatingList) == np.asarray(tempRatingList)[0]).all() \
                       or \
                       (np.asarray(tempGroupList) == np.asarray(tempGroupList[0])).all()
        # Now let’s calculate the pearson correlation between two users
        if len(tempRatingList) > 1 and len(tempGroupList) > 1 and not no_constants:
            corr = scipy.stats.pearsonr(tempRatingList, tempGroupList)[0]
            corr = float(corr)
            pearsonCorrelationDict[name] = corr
        else:
            pearsonCorrelationDict[name] = 0

    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['UserID'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))

    topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)
    topUsersRating = topUsers.merge(users_games_df, left_on='UserID', right_on='UserID', how='inner')

    topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']
    tempTopUsersRating = topUsersRating.groupby('GameID').sum()[['similarityIndex', 'weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']

    recommendation_df = pd.DataFrame()
    # Calculate the weighted average
    recommendation_df['weighted rec. score'] = tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating[
        'sum_similarityIndex']
    recommendation_df['GameID'] = tempTopUsersRating.index
    recommendation_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    recommendation_df = recommendation_df.dropna()
    recommendation_df = recommendation_df.sort_values(by='weighted rec. score', ascending=False)
    return recommendation_df


# Each count point is a match of suggested game
# with a game which was already played by the player
count = 0
n = 50
# Testing for users IDs from 1 to 100
for i in range(1, n):
    inputUser = i
    inputPreferences = users_games_df[users_games_df['UserID'] == inputUser]
    inputPreferences = inputPreferences.drop(columns=['UserID'])

    recommendation_df = calculateRecommendations(inputPreferences)

    # if we can not recommend, we just skip
    if len(recommendation_df) == 0:
        continue

    values = recommendation_df.head(3).astype('int64').values

    for j in range(len(values)):
        if recommendation_df.head(3).astype('int64').values[j][1] in (inputPreferences['GameID'].tolist()):
            count += 1
            # if our top-3 recommendation is in user's already played games,
            # then it is +1 point

print(f"self-defined accuracy = ", count / n)
