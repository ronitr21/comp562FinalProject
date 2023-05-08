from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
import tensorflow as tf
import keras
import math
import numpy as np
import pandas as pd
import makedata as mdata
import bracket
import bracket_year


print(tf.version.VERSION)


def makeModel(input_len, output_len):

    C = 200
    model = svm.SVC(kernel='rbf', probability=True)  # , verbose=True) #, max_iter=10_000)

    return model


def train(model, m, training_inputs, training_outputs, testing_inputs, testing_outputs):
    print('SHAPES')
    print(training_inputs.shape, training_outputs.shape,
          testing_inputs.shape, testing_outputs.shape)

    model = model.fit(training_inputs, training_outputs.flatten())
    print('MODEL IS FIT')

    wTRank = []
    lTRank = []
    wLRank = []
    lLRank = []
    wGTRank = []
    lGTRank = []
    wGLRank = []
    lGLRank = []
    wTWinTen = []
    lTWinTen = []
    wSWinTen = []
    lSWinTen = []
    wTWinZero = []
    lTWinZero = []
    wSWinZero = []
    lSWinZero = []
    didWin = []

    for i in range(0):  # len(training_inputs)):
        index = 0

        wTWinTen.append(training_inputs[i][index])
        index += 1
        lTWinTen.append(training_inputs[i][index])
        index += 1
        wSWinTen.append(training_inputs[i][index])
        index += 1
        lSWinTen.append(training_inputs[i][index])
        index += 1
        wTWinZero.append(training_inputs[i][index])
        index += 1
        lTWinZero.append(training_inputs[i][index])
        index += 1
        wSWinZero.append(training_inputs[i][index])
        index += 1
        lSWinZero.append(training_inputs[i][index])
        index += 1
        didWin.append(training_outputs[i][0])
        index += 1

    return m, model


def predictGame(m, model, year):
    def predict(team1, team2):
        inputs = mdata.getInputs(m, year, team1, team2)

        won = model.predict_proba(inputs)[0][1]
        los = model.predict_proba(inputs)[0][0]
        prob = won / (won + los)
        return prob
        # return (math.tanh(30*(prob-0.5))+1)/2.0
        # return model.predict(inputs)[0][0]

    return predict


def convertTeamToStr(m):
    def convert(teamId):
        if teamId in m.teamid_to_teamname:
            return m.teamid_to_teamname[teamId]
        return teamId + '-unknown'

    return convert


def testStuff(training_inputs, training_outputs, transform):
    correct = 0
    wrong = 0

    for i in range(len(training_inputs)):
        h = transform(training_inputs[i][0], training_inputs[i][2])
        a = transform(training_inputs[i][1], training_inputs[i][3])

        if h > a and training_outputs[i][0] == 1 or h < a and training_outputs[i][0] == 0:
            correct += 1
        else:
            wrong += 1

    return correct, wrong


if __name__ == "__main__":
    m, training_inputs, training_outputs, test_inputs, test_outputs = mdata.getData()
    print(m)
    print(training_inputs.shape[1])
    print(training_outputs.shape[1])
    model = makeModel(training_inputs.shape[1], training_outputs.shape[1])

    _, model = train(model, m, training_inputs, training_outputs, test_inputs, test_outputs)


    b = bracket.Bracket(bracket_year.the2016Bracket, predictGame(m, model, 2016), convertTeamToStr(m))
    b.playTourne()
    b = bracket.Bracket(bracket_year.the2018Bracket, predictGame(m, model, 2018), convertTeamToStr(m))
    b.playTourne()
    b = bracket.Bracket(bracket_year.the2019Bracket, predictGame(m, model, 2019), convertTeamToStr(m))
    b.playTourne()

    b = bracket.Bracket(bracket_year.the2021Bracket, predictGame(m, model, 2021), convertTeamToStr(m))
    b.playTourne()

    b = bracket.Bracket(bracket_year.the2021SecondChanceBracket, predictGame(m, model, 2021), convertTeamToStr(m))
    b.playTourne()

    b = bracket.Bracket(bracket_year.the2022Bracket, predictGame(m, model, 2022), convertTeamToStr(m))
    b.playTourne()

