"""
This experiment corresponds to the article
    "The Big-2/ROSe Model of Online Personality -- Towards a Lightweight Set
    of Markers for Characterizing the Behavior of Social Platform Denizens" (under review)

which is an extension of
    "Predicting User Reactions to Twitter Feed Content based on
    Personality Type and Social Cues" (under review)

The extension consists of using some traits in F1 and F2 of a proposed model called Big-2.
The selection is based on a deviation measure.

Another experiment variation is the possible outcomes,
in the former experiment were values between 1 and 4,
and in the later values between 1 and 16.

"""

import pickle  # to save lists in files
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import svm
from sklearn.metrics import confusion_matrix  # to calculate TP, TN, FP, FN
from sklearn.naive_bayes import MultinomialNB
from datetime import date
import datetime, time
from sklearn.naive_bayes import ComplementNB
import re  # for regular expressions
import os.path  # to determine whether a file exists
import numpy
from sklearn import metrics
import functions
import constants
import \
    math  # to check lists' values to use them in classifiers
from sklearn.linear_model import LogisticRegression


"""  
    Next variables must be set to configure the experiment 
"""
inputFileWithUsersIDs = constants.inputFileWithUsersIDs
PROCESS_CLASSIFIER = True  # This is useful to make tests that not include classfiers
listWithClassifiersToUse = ['LogisticRegression', 'DecisionTreeClassifier', 'OneClassSVM', 'RandomForestClassifier',
                                'MultinomialNB', 'ComplementNB']
REALLY_PROCESS_CANDIDATE_IDs = True  # When it is False all the elements in listaIntervalosAProcesar are processed. Otherwise, only are processed those in candidate list.
CRITERION_SELECTION_LIST_DIF_TARGET = [1,2,3,4]  # Are four categories according to three threshols.
CRITERION_SELECTION_DIF_TARGET = None
BALANCE_TRAINING = True
BALANCE_TEST = False
# OneClass parameter
ONE_CLASS_TARGET = 4  # It is the target value
GAMMAparameterList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                       1]  # For OneClassSVM()
KernelOneCParameterList = ['rbf', 'linear']
mixedListKernelAndGammas = [{'kernel': 'linear'}] + functions.mixupKernelsAndGammas('rbf',
                                                                                      GAMMAparameterList)  # para linear no tiene sentido gamma por eso no hay que mezclarlo
# RandomForests parameters
RForestMinSamplesLeafParameterList = [1, 5, 10, 20]
RFnEstimatorsParameterList = [10, 50, 100]
RandomForestMixedParameterList = functions.mixUpLists('min_samples_leaf', RForestMinSamplesLeafParameterList,
                                                               'n_estimators', RFnEstimatorsParameterList)
# MultinomialNB parameters
MultinomialNBalphaParameterList = [0, 0.05, 0.1, 0.15, 0.2,
                                    100]  # [0, 0,25, 0.5, 0.75, 1] # [1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 8, 10, 20]
MultinomialNBfitPriorParameterList = [True, False]
MultinomialNBMixedParameterList = functions.mixUpLists('alpha', MultinomialNBalphaParameterList, 'fit_prior',
                                                                MultinomialNBfitPriorParameterList)
# ComplementNB parameter
ComplementNBalphaParameterList = [0, 0.05, 0.1, 0.15, 0.2, 10]
ComplementNBnormParameterList = [True, False]
ComplementNBMixedParameterList = functions.mixUpLists('alpha', ComplementNBalphaParameterList, 'norm',
                                                               ComplementNBnormParameterList)

DICT_FEATURES_POSITIONS = {
    '0-OCEAN': False,
    '1-interval': True,
    '2-receivedHashtags': True,
    '3-POSproportion': True,
    '4-NEGproportion': True}  # FEATURES sublist structure are like [[23, 0, 0, 0, 0], [23, 1, 0, 0, 1], [23, 0, 0, 1, 1], ...]

processingIntervalList = ['1-3500']  # For processing in batches ['1-399', '400-1000', '1001-1251', '1501-2000']
kThatRemenber = constants.kThatRemenber
breadthsOfRememberedInterval = ['12hour']  # The interval could be something from this: ['15min', '30min', '1hour']
LEARNING_PERCENTAGE = 90

# These dates are useful to calculate how many intervals has the whole dataset.
BEGINING_DATE_DATASET = date(2013, 7, 15)
END_DATE_DATASET = date(2015, 3, 25)

# -------- cuando se quieren elegir algunas configuraciones para correr.
runChosenConfigurations = True  # When it's True only runs configurations in chosenConfigurationsList
chosenConfigurationsList = [["MultinomialNB", "alpha=100, fit_prior=True"], ["ComplementNB", "alpha=0.1, norm=True"]]

dictIntervalAmountPerDay = { # this have to be according to 'breadthsOfRememberedInterval'
    '15min': 96,
    '30min': 48,
    '1hour': 24,
    '12hour': 2,
    '1day': 1
}

# Edit the function translateIntevalToNumericValue(...) for each new interval
oneHourTranslateToNumberIntervalDict = {
    # Classifier does not receive string values, thus the interval '1hour' is translated to a numeric value.
    '00:00:00-01:00:00': 0,
    '01:00:00-02:00:00': 1,
    '02:00:00-03:00:00': 2,
    '03:00:00-04:00:00': 3,
    '04:00:00-05:00:00': 4,
    '05:00:00-06:00:00': 5,
    '06:00:00-07:00:00': 6,
    '07:00:00-08:00:00': 7,
    '08:00:00-09:00:00': 8,
    '09:00:00-10:00:00': 9,
    '10:00:00-11:00:00': 10,
    '11:00:00-12:00:00': 11,
    '12:00:00-13:00:00': 12,
    '13:00:00-14:00:00': 13,
    '14:00:00-15:00:00': 14,
    '15:00:00-16:00:00': 15,
    '16:00:00-17:00:00': 16,
    '17:00:00-18:00:00': 17,
    '18:00:00-19:00:00': 18,
    '19:00:00-20:00:00': 19,
    '20:00:00-21:00:00': 20,
    '21:00:00-22:00:00': 21,
    '22:00:00-23:00:00': 22,
    '23:00:00-00:00:00': 23
}

dictFrom12hourIntervalToNumeric = {
    '00:00:00-12:00:00': 0,
    '12:00:00-00:00:00': 1
}

OUTPUT_FILE_PATH                   = "Big2/classifiers/Outputs/"
FEATURES_OUTPUT_LIST_PATH          = "TargetsAndFeaturesLists/lists-"+str(breadthsOfRememberedInterval[0])+"-k"+str(kThatRemenber[0])+"/"  # this folder will contain *.py files containing features and targets in list format. The name of each file will be like "userID-k-interval.py"
CANDIDATE_IDs_FILE_PATH            = "Big2/classifiers/Outputs/candidatesPerCategory/categoryCandidateIDs" # to this path will be added  1,2,3, or 4 according to its category
OUTPUT_FILES_PER_CLASSIF_PATH      = "Big2/classifiers/clasOutputSummary(indiv)/"
SUMMARY_OUTPUT_FILE_PATH           = "Big2/classifiers/clasOutputSummary(indiv)/trainingDataSummaryr-" + constants.OCEAN_TYPE + "-k" + str(kThatRemenber[0]) + ".txt"

beginTime = datetime.datetime.now()

auxClassSummaryFileAndPathString = OUTPUT_FILE_PATH + constants.OCEAN_TYPE_STRING +"-SummaryClassif-"+str(breadthsOfRememberedInterval[0])+",k"+str(kThatRemenber[0])+".txt"
allClassifierSummaryOutputFile = open(auxClassSummaryFileAndPathString, "w")
summaryOutputFileForTraining = open(SUMMARY_OUTPUT_FILE_PATH, "w")

[keyNamesString, valueNamesString] = functions.convertKeysValuesDictIntoString(
    DICT_FEATURES_POSITIONS, '\t', '\t')
allClassHeaderSummaryString = (
        "Process candidate IDs?\tCandidate Criterion\tBalanceTraining?\tBalance Test?\t" + keyNamesString + "\tParameter\t\t" +
        "users' (position) list \t#users\tinterval\t#remember k\t" +
        "All FEATURES?\tClassifier algorithm\t#samples for trainingr\t" +
        "#samples for testing\t #average Training Features(#indivual elements/#Features)\t" +
        "#average Test Features(#individual elements/#Features)\t% for training\tScore_samples\t" +
        "#Inliers(1)\t#Outliers(-1)\tAccuracy\tPrecision\tRecall\tF1(lib)\tF1-F1previous\tF1ant / F1\tF1(own)\t#TN\t#FP\t#FN\t#TP\texecution time\t\tExtra info\n")

if os.path.getsize(auxClassSummaryFileAndPathString) == 0:
    # It's empty, so columns' header are added
    allClassifierSummaryOutputFile.write(allClassHeaderSummaryString)

dictIndividualFilesPerClass = functions.generateFilesForEachStringInList(listWithClassifiersToUse,
                                                                                        OUTPUT_FILES_PER_CLASSIF_PATH,
                                                                                        str(
                                                                                            breadthsOfRememberedInterval[
                                                                                                0]),
                                                                                        str(kThatRemenber[0]),
                                                                                        allClassHeaderSummaryString)

candidateIDsList = None
""" End of configuration variables """


def saveFeaturesAndTargetList(userID, FeatureWithOCEANlist, targetList, kValue, interval):
    """Given a user ID, a k value, an interval y two lists (featuresWithOcean and Target) save those lists en *.py files"""
    FeatureWithOCEANoutputFile = open(
        FEATURES_OUTPUT_LIST_PATH + userID + "-" + str(kValue) + "-" + interval + "-featureWithOCEAN.py",
        "wb")  # The file name will be like "userID-k-interval-featureWithOCEAN.py"
    pickle.dump(FeatureWithOCEANlist, FeatureWithOCEANoutputFile)

    outputTargetFile = open(FEATURES_OUTPUT_LIST_PATH + userID + "-" + str(kValue) + "-" + interval + "-target.py",
                            "wb")  # The file name will be like "userID-k-interval-target.py"
    pickle.dump(targetList, outputTargetFile)


def readListFeaturesAndTarget(userID, kValue, interval):
    """
    Given a user ID, k value and an interval returns three lists: feature list with and without OCEAN, and target
    """
    featureWithOCEANoutputList = None
    targetOutputList = None

    featureWithOCEANInputFile = open(
        FEATURES_OUTPUT_LIST_PATH + userID + "-" + str(kValue) + "-" + interval + "-featureWithOCEAN.py",
        "rb")  # The file name will be like "userID-k-interval-featureConOCEAN.py"
    featureWithOCEANoutputList = pickle.load(featureWithOCEANInputFile)

    targetInputFile = open(
        FEATURES_OUTPUT_LIST_PATH + userID + "-" + str(kValue) + "-" + interval + "-target.py",
        "rb")  # The file name will be like "userID-k-interval-target.py"
    targetOutputList = pickle.load(targetInputFile)

    return featureWithOCEANoutputList, targetOutputList


def main_process_users():
    position = 0
    FeatureWithOCEANlist = []
    FeatureWithoutOCEANlist = []
    TargetList = []
    successfulProcessedUserList = []  # each element of the list is like [pos,id] which is the position and user ID
    highestPositionToProcess = functions.getMaxValueAmongRanges(processingIntervalList)
    for oneLine in inputFileWithUsersIDs:
        position += 1
        if position <= highestPositionToProcess:
            userID = oneLine.replace('\n', '')
            if functions.belongToSomeRange(position,
                                                 processingIntervalList) and functions.existElementInList(
                    userID, candidateIDsList, REALLY_PROCESS_CANDIDATE_IDs):
                # oneLine is processed
                OCEANclass = None
                if constants.OCEAN_TYPE == "1024":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-1024.txt")
                if constants.OCEAN_TYPE == "32":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-32.txt")

                if constants.OCEAN_TYPE == "Mod2-1-1":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-16-Mod2-1-1.txt")
                if constants.OCEAN_TYPE == "Mod2-2-1":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-16-Mod2-2-1.txt")
                if constants.OCEAN_TYPE == "Mod2-3-1":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-16-Mod2-3-1.txt")
                if constants.OCEAN_TYPE == "Mod2-4-1":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-16-Mod2-4-1.txt")

                if constants.OCEAN_TYPE == "Mod2-1-2":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-16-Mod2-1-2.txt")
                if constants.OCEAN_TYPE == "Mod2-2-2":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-16-Mod2-2-2.txt")
                if constants.OCEAN_TYPE == "Mod2-3-2":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-16-Mod2-3-2.txt")
                if constants.OCEAN_TYPE == "Mod2-4-2":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-16-Mod2-4-2.txt")

                # New group with a higher deviation and OCEAN class between 1 and 4
                if constants.OCEAN_TYPE == "DesvioSinNyV":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-4-MayorDesvio-SinNyV.txt")

                if constants.OCEAN_TYPE == "DesvioMod2-1-2":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-4-Mod2-1-2.txt")
                if constants.OCEAN_TYPE == "DesvioMod2-2-2":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-4-Mod2-2-2.txt")
                if constants.OCEAN_TYPE == "DesvioMod2-3-2":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-4-Mod2-3-2.txt")
                if constants.OCEAN_TYPE == "DesvioMod2-4-2":
                    OCEANclass = functions.getOCEANarray(userID, "OCEAN-1-4-Mod2-4-2.txt")

                # It's verified whether exists the calculated OCEAN, thus those users without OCEAN are discarded
                if OCEANclass != None:
					# Given a user ID, it's obtained the file names with their full path according to interval breadths and k remembered.
                    for oneInterval in breadthsOfRememberedInterval:
                        for oneKvalue in kThatRemenber:
							if not (os.path.isfile(FEATURES_OUTPUT_LIST_PATH + userID + "-" + str(
                                    oneKvalue) + "-" + oneInterval + "-featureWithOCEAN.py") and os.path.isfile(
                                FEATURES_OUTPUT_LIST_PATH + userID + "-" + str(
                                    oneKvalue) + "-" + oneInterval + "-target.py")):
                                [answer, FeatureWithOCEANauxList,
                                 TargetAuxList] = processNewItemGivenUserBreadthIntervalAndK(userID, oneInterval, oneKvalue, OCEANclass)
                                if not (answer):
                                    print(" There is not user file ", userID, " with the interval: ",
                                          oneInterval, " and k: ", oneKvalue)
                                else:
                                    # the list is saved for its future usage
                                    saveFeaturesAndTargetList(userID, FeatureWithOCEANauxList, TargetAuxList,
                                                                 oneKvalue,
                                                                 oneInterval)

                                    FeatureWithOCEANlist = FeatureWithOCEANlist + FeatureWithOCEANauxList
                                    FeatureWithoutOCEANlist = FeatureWithoutOCEANlist + takeoutOCEANfromFeatureOCEANlist(FeatureWithOCEANauxList)
                                    TargetList = TargetList + TargetAuxList
                                    successfulProcessedUserList = successfulProcessedUserList + [
                                        [position, userID]]
                            else:
                                # Already exist two lists saved previously
								# Values from those lists are read for processing
                                FeatureWithOCEANauxList = functions.replaceValueInEachSublistGivenPosition(FeatureWithOCEANauxList, 0, OCEANclass)
                                FeatureWithoutOCEANAuxList = takeoutOCEANfromFeatureOCEANlist(FeatureWithOCEANauxList)
                                successfulProcessedUserList = successfulProcessedUserList + [
                                    [position, userID]]
                                FeatureWithOCEANlist = FeatureWithOCEANlist + FeatureWithOCEANauxList
                                FeatureWithoutOCEANlist = FeatureWithoutOCEANlist + FeatureWithoutOCEANAuxList
                                TargetList = TargetList + TargetAuxList
                else:
                    # The user does not have OCEAN
                    print("There is not  OCEAN value for the user " + str(userID) + " at position: " + str(
                        position) + "\n")
        else:
            # Max position was exceeded
            break
    # At this point all users were processed
    inputFileWithUsersIDs.seek(0)
    if FeatureWithOCEANlist != [] and FeatureWithoutOCEANlist != [] and TargetList != [] and PROCESS_CLASSIFIER:
        processWithClassifiersWithVaryingFeatures(successfulProcessedUserList, kThatRemenber[0], breadthsOfRememberedInterval[0], FeatureWithOCEANlist, TargetList)
    else:
        print("Classifiers are not processed")


def processNewItemGivenUserBreadthIntervalAndK(userID, oneInterval, kValue, OCEANclass):
    """
    Given a user, an interval and k value, process the corresponding news items file.
    Return false if there is not the file.
    """
    FeaturesWithOCEANlist = []  # This list will contain features with OCEAN, hashtag status, interval. For example [[0,...,1,...0, 0.3, 0.3, 0.4, 5],[],...]
    TargetsList = []

    answer = True
    intervalForFeature = ""
    newItemFileNameGivenIntervalAndK = getNewsItemsFilePathsGivenIDuser(userID, oneInterval, kValue)

    if os.path.isfile(newItemFileNameGivenIntervalAndK):
        newsItemInputFileGivenUserIDintervalK = open(newItemFileNameGivenIntervalAndK, "r", encoding='UTF8')

        newsItemAmountToProcess = countIntervalsGivenBreadth(oneInterval)
        readNewsItemsAmount = 0
        readLinesFileAmount = 0
        initialArrayNewsItem = ["", "", ""]
        arrayNewsItemsJoiningInitialsOneInterval = []  # is a list where each element is a sublist of [initialArrayNewsItem]
        showNewInterval = False
        joinedListAmount = 0
        notIgnoredNotEmptyAmount = 0
        try:
            for newsItemLine in newsItemInputFileGivenUserIDintervalK:
                readLinesFileAmount += 1
                newsItemLine = newsItemLine.replace("\n", "").replace('"', '')
                arrayNewsItem = newsItemLine.split('\t')

                if readLinesFileAmount == 1:
                    initialArrayNewsItem = arrayNewsItem
                isCutNewsItemList = False

                if arrayNewsItem[0] != "":
                    if arrayNewsItem[0][0] != '(':
                        # New interval of the newsitems
                        readNewsItemsAmount += 1

                        if initialArrayNewsItem[1][:7] != '<empty>' and initialArrayNewsItem[2][:7] != '<empty>':
                            showNewInterval = True
                    else:
                        isCutNewsItemList = True

                        initialArrayNewsItem[2] = initialArrayNewsItem[2] + ";" + arrayNewsItem[0]

                        joinedListAmount += 1

                if readNewsItemsAmount <= (
                        newsItemAmountToProcess + 1):
                    # the news item is processed
                    if not (isCutNewsItemList) and readNewsItemsAmount > 1:
                        valueToRest = 0
                        if showNewInterval:
                            valueToRest = 1

                        # if the interval is empty it is discarded
                        if initialArrayNewsItem[1][:7] != '<empty>' and initialArrayNewsItem[2][:7] != '<empty>':
                            arrayNewsItemsJoiningInitialsOneInterval = arrayNewsItemsJoiningInitialsOneInterval + [
                                initialArrayNewsItem]

                        initialArrayNewsItem = arrayNewsItem
                        joinedListAmount = 0

                        if showNewInterval and arrayNewsItemsJoiningInitialsOneInterval != []:
                            # new interval is read
                            intervalForFeature = getJustHourInterval((arrayNewsItemsJoiningInitialsOneInterval[0])[0])
                            # The interval format is like:  3349[2013-12-01 12:00:00-2013-12-01 13:00:00] thus with the function is extract '12:00:00-13:00:00'
                            [arrayHashtags, arrayTarget] = getNormalizedStatusHashtagAndTarget(
                                arrayNewsItemsJoiningInitialsOneInterval)

                            FeaturesWithOCEANlist = FeaturesWithOCEANlist + [[OCEANclass] + [
                                translateIntevalToNumericValue(intervalForFeature, oneInterval)] + arrayHashtags]
                            TargetsList = TargetsList + arrayTarget

                            arrayNewsItemsJoiningInitialsOneInterval = []

                            showNewInterval = False

                else:
                    break  # for avoiding keep reading lines unnecessary
        except UnicodeDecodeError as error:
            print("ERROR UnicodeDecodeError!!! userID: ", userID)
    else:
        # There is not file for given id-interval-k
        answer = False
        FeaturesWithOCEANlist = None
        TargetsList = None
    return answer, FeaturesWithOCEANlist, TargetsList


def getNormalizedStatusHashtagAndTarget(newsItemInputarrayWholeInterval):
    """
    #  Given a list of sublist containing the full status of one interval, each sublist is like:
    #   [0]: indicated the interval
    #   [1]: user decision,
    #   [2]: news item list of received news items (eventually empty)
    # Return two arrays:
    #       (1) An array with three positions:
    #           [0]: can be  '0', '1', o '2', indicating which is the preponderant sentiment in the interval: neu, pos or neg, resp.
    #       (2) target array containig a value between 0 and 4
    #           0: the user ignored the news item,
    #           1,2,3: did not make up, and whether reuse neg or neutro resp.
    #           4: the user made up something.
    """
    featureHashtagOutputArray = [0, 0, 0]
    targetOutputArray = [0]

    ignoredAmount = 0
    intervalAmount = 0
    reusePositiveAmount = 0
    reuseNegativeAmount = 0
    reuseNeutralAmount = 0
    sublistAmount = 0
    receivedPositiveAmount = 0
    receivedNegativeAmount = 0
    receivedNeutralAmount = 0
    for oneSublist in newsItemInputarrayWholeInterval:
        sublistAmount += 1
        if oneSublist[2][:11] != '<made up>':
            # at least there was one news item
            receivedNewsItemArray = oneSublist[2].split(';')
            newsItemAmount = len(receivedNewsItemArray)
            receivedPositiveAmountGivenHashtag = 0
            receivedNegativeAmountGivenHashtag = 0
            receivedNeutralAmountGivenHashtag = 0
            for oneNewsItemSublist in receivedNewsItemArray:
                sentiment = getNewsItemPortion(oneNewsItemSublist, 's')
                if sentiment == 'pos':
                    receivedPositiveAmountGivenHashtag += 1
                else:
                    if sentiment == 'neg':
                        receivedNegativeAmountGivenHashtag += 1
                    else:
                        if sentiment == 'neu':
                            receivedNeutralAmountGivenHashtag += 1

            receivedPositiveAmount = receivedPositiveAmount + receivedPositiveAmountGivenHashtag
            receivedNegativeAmount = receivedNegativeAmount + receivedNegativeAmountGivenHashtag
            receivedNeutralAmount = receivedNeutralAmount + receivedNeutralAmountGivenHashtag

            # one line processing has finiched (related to one hashtag)
            # the predominant sentiment is determined
            addedSentiments = determinePreponderantSentiment(receivedPositiveAmountGivenHashtag,
                                                             receivedNegativeAmountGivenHashtag,
                                                             receivedNeutralAmountGivenHashtag)

        if oneSublist[2][:11] == '<made up>':
            targetOutputArray = [4]
            intervalAmount += 1
        else:
            if oneSublist[1][:10] != '<ignored>':
                sentiment = getNewsItemPortion(oneSublist[1], 's')
                if sentiment == 'pos':
                    reusePositiveAmount += 1
                if sentiment == 'neg':
                    reuseNegativeAmount += 1
                if sentiment == 'neu':
                    reuseNeutralAmount += 1
            else:
                ignoredAmount += 1

    # all sublist are already processed
    addedSentimentAsIntervalSummary = determinePreponderantSentiment(receivedPositiveAmount, receivedNegativeAmount,
                                                                     receivedNeutralAmount)

    if addedSentimentAsIntervalSummary == 'neu':
        featureHashtagOutputArray[0] = 0
    else:
        if addedSentimentAsIntervalSummary == 'pos':
            featureHashtagOutputArray[0] = 1
        else:
            featureHashtagOutputArray[0] = 2  # 'neg' for default

    auxPos = 0
    auxNeg = 0
    if (receivedPositiveAmount + receivedNegativeAmount + receivedNeutralAmount) > 0:
        auxPos = ((receivedPositiveAmount * 100) / (
                receivedPositiveAmount + receivedNegativeAmount + receivedNeutralAmount)) / 100
        auxNeg = ((receivedNegativeAmount * 100) / (
                    receivedPositiveAmount + receivedNegativeAmount + receivedNeutralAmount)) / 100
    featureHashtagOutputArray[1] = translatePercentageToNumberGivenInterval(auxPos)
    featureHashtagOutputArray[2] = translatePercentageToNumberGivenInterval(auxNeg)

    if intervalAmount == 0:
        # Any made up news item in the interval
        if (reusePositiveAmount + reuseNeutralAmount + reuseNegativeAmount) != 0:
            #  at least there was a reused
            porcentajeNIsPosEnTodoElIntervalo = (reusePositiveAmount * 100) / (
                        reusePositiveAmount + reuseNeutralAmount + reuseNegativeAmount)
            porcentajeNIsNegEnTodoElIntervalo = (reuseNegativeAmount * 100) / (
                        reusePositiveAmount + reuseNeutralAmount + reuseNegativeAmount)
            porcentajeNIsNeuEnTodoElIntervalo = (reuseNeutralAmount * 100) / (
                        reusePositiveAmount + reuseNeutralAmount + reuseNegativeAmount)
        else:
            # any reuse (or did nothing)
            targetOutputArray = [0]

    return featureHashtagOutputArray, targetOutputArray


def determinePreponderantSentiment(positiveAmount, negativeAmount, negativeAmount):
    """
    Given three values corresponding to positive, negative and neutral amounts, returns predominant sentiment
    """
    winningSentiment = None
    if positiveAmount > negativeAmount and positiveAmount > negativeAmount:
        # 'pos' wins
        winningSentiment = 'pos'
    else:
        if negativeAmount > positiveAmount and negativeAmount > negativeAmount:
            # 'neg' wins
            winningSentiment = 'neg'
        else:
            if negativeAmount > positiveAmount and negativeAmount > negativeAmount:
                # 'neu' wins
                winningSentiment = 'neu'
            else:
                # there is a tie
                if positiveAmount == negativeAmount and negativeAmount == negativeAmount:
                    # tie between the three sentiments, thus we asumme that 'neu' wins
                    winningSentiment = 'neu'
                else:
                    # there is a tie between two
                    if positiveAmount == negativeAmount:
                        # tie between 'pos' and 'neg'. We assume 'neu' wins
                        winningSentiment = 'neu'
                    else:
                        # pos o neg tie with neu
                        if positiveAmount > negativeAmount:
                            winningSentiment = 'pos'
                        else:
                            winningSentiment = 'neg'

    return winningSentiment


def translatePercentageToNumberGivenInterval(percentage):
    """
    Given a percentage in [0,1], it's translate into a value in {0,1,2,3} accorsing to [0, 0.25), [0.25, 0.5), [0.5, 0,75) o [0,75,1]
    """
    answer = None
    if percentage < 0.25:
        answer = 0
    else:
        if percentage < 0.5:
            answer = 1
        else:
            if percentage < 0.75:
                answer = 2
            else:
                if percentage <= 1:
                    answer = 3
    return answer


def getNewsItemPortion(newsItem, element):
    """
    Given a news item as string return a part of ir according to
    'element':
         'o': origin,
         'l': hashtag/literal,
         's': sentiment,
         'f': date,
         'h': time,
         'fh': date and time, for instance '2014-05-10 11:09:01'.
    """
    answer = ""
	newsItem = (newsItem[:-1])[1:]  # the symbols (  at the beginning and ] at the end are discarded
    arrayNI = newsItem.split(')[')
    arrayNI = arrayNI[0].split(',') + arrayNI[1].split(' ')

    if element == 'o':
        answer = arrayNI[0]
    else:
        if element == 'l':
            answer = arrayNI[1]
        else:
            if element == 's':
                answer = arrayNI[2]
            else:
                if element == 'f':
                    answer = arrayNI[3]
                else:
                    if element == 'h':
                        answer = arrayNI[4]
                    else:
                        if element == 'fh':
                            answer = arrayNI[3] + " " + arrayNI[4]
                        else:
                            answer = "ERROR"
    return answer


def getJustHourInterval(fullInterval):
    """
    Given an inteval as evalution format of news item like  "3349[2013-12-01 12:00:00-2013-12-01 13:00:00]"
    return only the beginning and end hour (according to the example, '12:00:00-13:00:00')
    """
    aux = (fullInterval.split('[')[1]).split(
        ' ')  # this is something like ['2013-12-01', '12:00:00-2013-12-01', '13:00:00]']
    return (aux[1].split('-'))[0] + "-" + (aux[2])[:-1]


def getNewsItemsFilePathsGivenIDuser(id, intervalTimeBreadth, kValue):
    newsItemFileBasicPath = "..."
    return newsItemFileBasicPath + str(intervalTimeBreadth) + "-k" + str(kValue) + "/" + str(
        id) + "-" + str(intervalTimeBreadth) + "-k" + str(kValue) + ".txt"


def countIntervalsGivenBreadth(breadth):
    """
    Given a breadth, return how many time it fits in the dataset duration
    """
    return ((END_DATE_DATASET - BEGINING_DATE_DATASET).days + 1) * dictIntervalAmountPerDay[breadth]


def translateIntevalToNumericValue(interval, breadth):
    answer = None
    if breadth == '1hour':
        answer = oneHourTranslateToNumberIntervalDict[interval]
    if breadth == '12hour':
        answer = dictFrom12hourIntervalToNumeric[interval]
    return answer


def takeoutOCEANfromFeatureOCEANlist(featureWithOCEANlist):
    featureWithoutOCEANlist = [sublista[1:] for sublista in featureWithOCEANlist]
    return featureWithoutOCEANlist


def processWithClassifiersWithVaryingFeatures(processUsersList, kValue, interval, fullFeaturesList,
                                                    targetsList):
    """
    Given input lists, evaluate with classifiers and the result are save in files
    """
    featuresWithoutOCEANlist = functions.activateDeativateFeatures(fullFeaturesList, DICT_FEATURES_POSITIONS)

    processedUsersString = ""
    for oneElement in processUsersList:
        processedUsersString += "(" + str(oneElement[0]) + ")" + str(oneElement[1]) + ","

    listLength = len(fullFeaturesList)
    percentageLimitIndex = int((listLength * LEARNING_PERCENTAGE) / 100)
    featuresWithOCEANForTrainingList = fullFeaturesList[0:percentageLimitIndex]
    featuresWithOCEANForTestingList = fullFeaturesList[percentageLimitIndex:listLength]
    featuresWithoutOCEANForTrainingList = featuresWithoutOCEANlist[0:percentageLimitIndex]
    featuresWithoutOCEANforTestingList = featuresWithoutOCEANlist[percentageLimitIndex:listLength]
    targetsForTrainingList = targetsList[0:percentageLimitIndex]
    targetsForTestingList = targetsList[percentageLimitIndex:listLength]

    if BALANCE_TEST:
        [featuresWithOCEANForTestingList, targetForTestingListNoUsefulHere] = functions.balanceTargets(
            featuresWithOCEANForTestingList,
            targetsForTestingList)
        [featuresWithoutOCEANforTestingList, targetsForTestingList] = functions.balanceTargets(
            featuresWithoutOCEANforTestingList, targetsForTestingList)

    if BALANCE_TRAINING:
        [featuresWithOCEANForTrainingList, targetsForTraningListNoUsefulHere] = functions.balanceTargets(
            featuresWithOCEANForTrainingList,
            targetsForTrainingList)
        [featuresWithoutOCEANForTrainingList, targetsForTrainingList] = functions.balanceTargets(
            featuresWithoutOCEANForTrainingList, targetsForTrainingList)

    for oneClasifier in listWithClassifiersToUse:
        if oneClasifier == "LogisticRegression" or oneClasifier == "DecisionTreeClassifier":
            f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithOCEANForTrainingList,
                                                                        targetsForTrainingList,
                                                                        featuresWithOCEANForTestingList,
                                                                        targetsForTestingList, True,
                                                                        processedUsersString,
                                                                        len(processUsersList), interval,
                                                                        kValue, oneClasifier, paramConstructor=0,
                                                                        previousf1=0)
            f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithoutOCEANForTrainingList,
                                                                        targetsForTrainingList,
                                                                        featuresWithoutOCEANforTestingList,
                                                                        targetsForTestingList, False,
                                                                        processedUsersString,
                                                                        len(processUsersList), interval,
                                                                        kValue, oneClasifier, paramConstructor=0,
                                                                        previousf1=f1Aux)

        if oneClasifier == 'OneClassSVM':
            # oneClass
            for oneConfigWithParameters in mixedListKernelAndGammas:
                f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithOCEANForTrainingList,
                                                                            targetsForTrainingList,
                                                                            featuresWithOCEANForTestingList,
                                                                            targetsForTestingList, True,
                                                                            processedUsersString,
                                                                            len(processUsersList), interval,
                                                                            kValue, oneClasifier,
                                                                            paramConstructor=oneConfigWithParameters,
                                                                            previousf1=0)
                f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithoutOCEANForTrainingList,
                                                                            targetsForTrainingList,
                                                                            featuresWithoutOCEANforTestingList,
                                                                            targetsForTestingList, False,
                                                                            processedUsersString,
                                                                            len(processUsersList), interval,
                                                                            kValue, oneClasifier,
                                                                            paramConstructor=oneConfigWithParameters,
                                                                            previousf1=f1Aux)

        if oneClasifier == 'RandomForestClassifier':
            for oneConfigurationRF in RandomForestMixedParameterList:
                f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithOCEANForTrainingList,
                                                                            targetsForTrainingList,
                                                                            featuresWithOCEANForTestingList,
                                                                            targetsForTestingList, True,
                                                                            processedUsersString,
                                                                            len(processUsersList), interval,
                                                                            kValue, oneClasifier,
                                                                            paramConstructor=oneConfigurationRF, previousf1=0)
                f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithoutOCEANForTrainingList,
                                                                            targetsForTrainingList,
                                                                            featuresWithoutOCEANforTestingList,
                                                                            targetsForTestingList, False,
                                                                            processedUsersString,
                                                                            len(processUsersList), interval,
                                                                            kValue, oneClasifier,
                                                                            paramConstructor=oneConfigurationRF,
                                                                            previousf1=f1Aux)
        if oneClasifier == 'MultinomialNB':
            for oneConfigMBN in MultinomialNBMixedParameterList:
                summaryOutputFileForTraining.write(
                    "\n\t\t\t Classifier: " + str(oneClasifier) + "     One config: " + str(
                        oneConfigMBN) + "   CRITERION_SELECTION_DIF_TARGET: " + str(
                        CRITERION_SELECTION_DIF_TARGET) + "\n")

                f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithOCEANForTrainingList,
                                                                            targetsForTrainingList,
                                                                            featuresWithOCEANForTestingList,
                                                                            targetsForTestingList, True,
                                                                            processedUsersString,
                                                                            len(processUsersList), interval,
                                                                            kValue, oneClasifier,
                                                                            paramConstructor=oneConfigMBN, previousf1=0)
                f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithoutOCEANForTrainingList,
                                                                            targetsForTrainingList,
                                                                            featuresWithoutOCEANforTestingList,
                                                                            targetsForTestingList, False,
                                                                            processedUsersString,
                                                                            len(processUsersList), interval,
                                                                            kValue, oneClasifier,
                                                                            paramConstructor=oneConfigMBN,
                                                                            previousf1=f1Aux)
        if oneClasifier == 'ComplementNB':
            for oneConfigComplmentBN in ComplementNBMixedParameterList:
                summaryOutputFileForTraining.write(
                    "\n\t\t\t Classifier: " + str(oneClasifier) + "     one config: " + str(
                        oneConfigComplmentBN) + "   CRITERION_SELECTION_DIF_TARGET: " + str(
                        CRITERION_SELECTION_DIF_TARGET) + "\n")

                f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithOCEANForTrainingList,
                                                                            targetsForTrainingList,
                                                                            featuresWithOCEANForTestingList,
                                                                            targetsForTestingList, True,
                                                                            processedUsersString,
                                                                            len(processUsersList), interval,
                                                                            kValue, oneClasifier,
                                                                            paramConstructor=oneConfigComplmentBN,
                                                                            previousf1=0)
                f1Aux = processWithClassifierOneClassDecisionTreeLogistic(featuresWithoutOCEANForTrainingList,
                                                                            targetsForTrainingList,
                                                                            featuresWithoutOCEANforTestingList,
                                                                            targetsForTestingList, False,
                                                                            processedUsersString,
                                                                            len(processUsersList), interval,
                                                                            kValue, oneClasifier,
                                                                            paramConstructor=oneConfigComplmentBN,
                                                                            previousf1=f1Aux)

        dictIndividualFilesPerClass[str(oneClasifier)].write(
            "-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n")
    allClassifierSummaryOutputFile.write(
        "-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n")
    return True


def processWithClassifierOneClassDecisionTreeLogistic(featuresForTrainingList,
                                                      targetsForTrainingList, featuresForTestingList,
                                                      targetsForTestingList,
                                                      allFeatures, processedUsersList,
                                                      processedUsersAmount,
                                                      interval, kValue, classifierType, paramConstructor,
                                                      previousf):
    """Return f1"""
    beginningTimeOneClass = datetime.datetime.now()
    constructorParameterString = "<no apply>"
    allFeaturesString = ""
    if allFeatures:
        allFeaturesString = "all"
    else:
        allFeaturesString = "some"

    clf = None
    if classifierType == "LogisticRegression":
        clf = LogisticRegression()
    else:
        if classifierType == "DecisionTreeClassifier":
            clf = tree.DecisionTreeClassifier()
        else:
            if classifierType == "OneClassSVM":
                if paramConstructor['kernel'] == 'rbf':
                    clf = svm.OneClassSVM(gamma=paramConstructor['gamma'],
                                          kernel='rbf')
                    constructorParameterString = "kernel=rbf" + ", gamma=" + str(paramConstructor['gamma'])
                else:
                    if paramConstructor['kernel'] == 'linear':
                        clf = svm.OneClassSVM(
                            kernel='linear')
                        constructorParameterString = "kernel=linear"
                    else:
                        constructorParameterString = "Kernel is neither rbf nor linear"
            else:
                if classifierType == "RandomForestClassifier":
                    clf = RandomForestClassifier(min_samples_leaf=int(paramConstructor['min_samples_leaf']),
                                                 n_estimators=int(paramConstructor['n_estimators']))
                    constructorParameterString = "min_samples_leaf=" + str(
                        paramConstructor['min_samples_leaf']) + ", n_estimators" + str(paramConstructor['n_estimators'])
                else:
                    if classifierType == 'MultinomialNB':
                        clf = MultinomialNB(alpha=float(paramConstructor['alpha']),
                                            fit_prior=paramConstructor['fit_prior'])
                        constructorParameterString = "alpha=" + str(paramConstructor['alpha']) + ", fit_prior=" + str(
                            paramConstructor['fit_prior'])

                    else:
                        if classifierType == 'ComplementNB':
                            clf = ComplementNB(alpha=float(paramConstructor['alpha']), norm=paramConstructor['norm'])
                            constructorParameterString = "alpha=" + str(
                                paramConstructor['alpha']) + ", norm=" + str(paramConstructor['norm'])

    f1 = -1
    if not (runChosenConfigurations) or [classifierType,
                                         constructorParameterString] in chosenConfigurationsList:
        startFitTime = datetime.datetime.now()

        [featuresCertainTargetsForTrainingList,
         targetCertaingTargetsForTrainingList] = functions.getFeaturesAndTargetsAccordingToTargetValue(
            featuresForTrainingList, targetsForTrainingList, ONE_CLASS_TARGET, True)

        if classifierType == "OneClassSVM":
            clf.fit(featuresCertainTargetsForTrainingList)
        else:
            print("\t CRITERION_SELECTION_DIF_TARGET: ", CRITERION_SELECTION_DIF_TARGET, "\n candidateIDsList",
                  candidateIDsList, "\n featuresForTrainingList:\n", featuresForTrainingList,
                  "\n\t targetsForTrainingList:\n", targetsForTrainingList)
            clf.fit(featuresForTrainingList, targetsForTrainingList)

            summaryOutputFileForTraining.write(
                "Classifier type: " + classifierType + "\n FIT:\n\t Training List" + "\n")
            summaryOutputFileForTraining.write(str(featuresForTrainingList))
            summaryOutputFileForTraining.write("\n\tTraining List:\n")
            summaryOutputFileForTraining.write(str(targetsForTrainingList))

        # RECALL and PRECISION
        predictAnswer = clf.predict(featuresForTestingList)

        summaryOutputFileForTraining.write("\n\t Prediction list:" + "\n")
        summaryOutputFileForTraining.write(str(featuresForTestingList))
        summaryOutputFileForTraining.write("\n")
        summaryOutputFileForTraining.write("\n\t\tpredict answer before replacemnt \n" + str(predictAnswer))
        summaryOutputFileForTraining.write("\n\n\n")

        predictAnswer = functions.replaceValuesInList(predictAnswer, ONE_CLASS_TARGET, 1, True)
        predictAnswer = functions.replaceValuesInList(predictAnswer, 1, -1, False)

        summaryOutputFileForTraining.write("\n\t\tPredict answer after remplacement \n" + str(predictAnswer))
        summaryOutputFileForTraining.write("\n\n\n")

        scores = "<Without scores>"
        if classifierType == "OneClassSVM":
            scores = clf.score_samples(featuresForTestingList)
        else:
            scores = clf.score(featuresForTestingList, targetsForTestingList)

        predictInliersAmount = len([i for i, e in enumerate(predictAnswer) if e == 1])
        predictOutliersAmount = len([i for i, e in enumerate(predictAnswer) if e == -1])

        # 1: inliers, -1: outliers
        targs = functions.replaceValuesInList(targetsForTestingList, ONE_CLASS_TARGET, 1, True)
        targs = functions.replaceValuesInList(targs, 1, -1, False)

        [accuracy, precision, recall, f1] = ["<without valor>", "<without valor>", "<without valor>", "<without valor>"]
        if classifierType == "OneClassSVM":
            [accuracy, precision, recall, f1] = functions.calculateAccPrecRecF1(targetsForTestingList, predictAnswer,
                                                                                True, ONE_CLASS_TARGET)
        else:
            [accuracy, precision, recall, f1] = functions.calculateAccPrecRecF1(targetsForTestingList, predictAnswer,
                                                                                True, ONE_CLASS_TARGET)

        [featuresCertainTargetList,
         targetsCertainTargetList] = functions.getFeaturesAndTargetsAccordingToTargetValue(featuresForTrainingList,
                                                                                           targetsForTrainingList,
                                                                                           ONE_CLASS_TARGET, True)
        X_train = featuresCertainTargetList
        X_test = featuresForTestingList
        [featuresNotCertainTargetList,
         targetsNotCertainTargetList] = functions.getFeaturesAndTargetsAccordingToTargetValue(featuresForTrainingList,
                                                                                              targetsForTrainingList,
                                                                                              ONE_CLASS_TARGET,
                                                                                              False)
        X_outliers = featuresNotCertainTargetList

        #   - M E T R I C S -

        avgColumnTrainingFeaturesString = functions.getAvgElementsSublists(featuresForTrainingList, True)
        avgTestingFeaturesColumnString = functions.getAvgElementsSublists(featuresForTestingList, True)

        inlierAmountString = str(predictInliersAmount) + "(" + str(
            round(predictInliersAmount / len(predictAnswer), 2)) + "%)"
        outliersAmountString = str(predictOutliersAmount) + "(" + str(
            round(predictOutliersAmount / len(predictAnswer), 2)) + "%)"

        [stringTN, stringFP, stringFN, stringTP] = functions.calculateTPTNFPFN(predictAnswer, targs, [-1, 1], True)

        summaryOutputFileForTraining.write("\n\t\t\t ---------------------- ")
        summaryOutputFileForTraining.write("\n\t\t\t stringTN: " + stringTN + "\n")
        summaryOutputFileForTraining.write("\n\t\t\t stringFP: " + stringFP + "\n")
        summaryOutputFileForTraining.write("\n\t\t\t stringFN: " + stringFN + "\n")
        summaryOutputFileForTraining.write("\n\t\t\t stringTP: " + stringTP + "\n")
        summaryOutputFileForTraining.write("\t\t\t ---------------------- \n")

        endTimeOneClass = datetime.datetime.now()
        oneClassExecutionClass = str(endTimeOneClass - beginningTimeOneClass)

        stringAuxDivF1 = "0"
        stringAuxRestaF1 = ""
        if not (allFeatures):
            stringAuxRestaF1 = str(f1 - previousf)

            if f1 != 0:
                stringAuxDivF1 = str(previousf / f1)

        allClassifierSummaryOutputFile.write(
            str(REALLY_PROCESS_CANDIDATE_IDs) + '\t' + "Dif of " + str(CRITERION_SELECTION_DIF_TARGET) + '\t' + str(
                BALANCE_TRAINING) + '\t' +
            str(BALANCE_TEST) + "\t" + valueNamesString + "\t" + constructorParameterString + '\t\t')

        allClassifierSummaryOutputFile.write(
            processedUsersList + '\t' + str(processedUsersAmount) + '\t' + interval + '\t' +
            str(kValue) + '\t' + allFeaturesString + '\t' + classifierType + '\t' +
            str(len(featuresForTrainingList)) + '\t' + str(len(featuresForTestingList)) + '\t' +
            avgColumnTrainingFeaturesString + '\t' +
            avgTestingFeaturesColumnString + '\t' + str(LEARNING_PERCENTAGE) + '\t' +
            (functions.concatListIntoString(scores, ', '))[:64] + '\t' + inlierAmountString + '\t' +
            outliersAmountString + '\t' + str(accuracy) + '\t' + str(precision) + '\t' +
            str(recall) + '\t' + str(f1) + '\t' + stringAuxRestaF1 + '\t' + stringAuxDivF1 + '\t' + str(
                (precision + recall) / 2) + '\t' +
            stringTN + '\t' + stringFP + '\t' + stringFN + '\t' + stringTP + '\t' +
            oneClassExecutionClass + '\t\t' + + '\n')

        dictIndividualFilesPerClass[str(classifierType)].write(
            str(REALLY_PROCESS_CANDIDATE_IDs) + '\t' + "Dif of " + str(CRITERION_SELECTION_DIF_TARGET) + '\t' + str(
                BALANCE_TRAINING) + '\t' +
            str(BALANCE_TEST) + "\t" + valueNamesString + "\t" + constructorParameterString + '\t\t')
        dictIndividualFilesPerClass[str(classifierType)].write(
            processedUsersList + '\t' + str(processedUsersAmount) + '\t' + interval + '\t' +
            str(kValue) + '\t' + allFeaturesString + '\t' + classifierType + '\t' +
            str(len(featuresForTrainingList)) + '\t' + str(len(featuresForTestingList)) + '\t' +
            avgColumnTrainingFeaturesString + '\t' +
            avgTestingFeaturesColumnString + '\t' + str(LEARNING_PERCENTAGE) + '\t' +
            (functions.concatListIntoString(scores, ', '))[:64] + '\t' + inlierAmountString + '\t' +
            outliersAmountString + '\t' + str(accuracy) + '\t' + str(precision) + '\t' +
            str(recall) + '\t' + str(f1) + '\t' + stringAuxRestaF1 + '\t' + stringAuxDivF1 + '\t' + str(
                (precision + recall) / 2) + '\t' +
            stringTN + '\t' + stringFP + '\t' + stringFN + '\t' + stringTP + '\t' +
            oneClassExecutionClass + '\t\t' + + '\n')
    return f1


"""
    End of functions' declarations 
"""
for oneCriterion in CRITERION_SELECTION_LIST_DIF_TARGET:
    CRITERION_SELECTION_DIF_TARGET = oneCriterion
    # The users' IDs file is opened
    if PROCESS_CLASSIFIER:
        candidateIDsList = functions.generateListFromFile(
            open(CANDIDATE_IDs_FILE_PATH + str(oneCriterion) + ".txt", "r"))
    if oneCriterion == 100:
        REALLY_PROCESS_CANDIDATE_IDs = False  # because 100 is not used as a creterion, thus all users are considered
    main_process_users()


finishTime = datetime.datetime.now()
print("BEGINNING - FINISH time: \t ", beginTime, " \n\t\t\t\t\t\t ", finishTime)
print("Duration: ", finishTime - beginTime)
