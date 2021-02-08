import constants
import os.path
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import collections
import numpy as np
import pickle  # to save lists in files


def readListFeaturesAndTarget(userID, valueK, range, pathListsFeatTtarg):
    ouputListFeatureWithOcean = None
    outputListTarget = None

    inputFileFeautureWithOcean = open(
        pathListsFeatTtarg + userID + "-" + str(valueK) + "-" + range + "-featureWithOCEAN.py",
        "rb")  # the name of the file will be like "userID-k-range-featureWithOCEAN.py"
    ouputListFeatureWithOcean = pickle.load(inputFileFeautureWithOcean)

    inputFileTarget = open(
        pathListsFeatTtarg + userID + "-" + str(valueK) + "-" + range + "-target.py",
        "rb")  # the name of the file will be like "userID-k-range-target.py"
    outputListTarget = pickle.load(inputFileTarget)

    return ouputListFeatureWithOcean, outputListTarget


def belongToSomeRange(value, intList):
    """
    Given a int value evaluates whether is or not in the range list
    List example ['5-20', '40-60', '90-95']
    """
    answer = False
    value = int(value)
    for oneRange in intList:
        arrayAux = oneRange.split('-')
        if value >= int(arrayAux[0]) and value <= int(arrayAux[1]):
            answer = True
            break
    return answer


def getMaxValueAmongRanges(intList):
    """
    Given a range list, returns the max value contained in one of them
    For example, given ['5-20', '40-60', '90-95'] --> returns 95
    """
    answer = 0
    for oneRange in intList:
        arrayAux = oneRange.split('-')
        if int(arrayAux[1]) > answer:
            answer = int(arrayAux[1])
    return answer


def generateListFromFile(inputFile):
    """
    Given a file, generates a list with its elements
    """
    outputList = []
    for oneLine in inputFile:
        outputList += [(oneLine.split('\t'))[0]]
    return outputList


def existElementInList(elem, inputList, processOrNot):
    """
    Returns True or False whether an element exists in a list
    The parameter processOrNot returns False no matter what. Otherwise, evaluates as described before.
    """
    answer = False
    if processOrNot:
        answer = elem in inputList
    else:
        answer = True
    return answer


def balanceTargets(featuresList, targetsList):
    """
    Given two lists, from the target list are count the '4' values, and the
    same amount of target '0' is chosen randomly.
    """
    featuresOutputList = []
    outputTargetsList = []

    positionsList4 = [i for i, e in enumerate(targetsList) if
                      e == 4]  # contains a list of positions that contain '4' as value in targetsList.
    positionsListNot4 = [i for i, e in enumerate(targetsList) if
                         e != 4]  # contains a list of positions that NOT contain '4' as value in targetsList.

    # the amount of 4's and not 4's are balanced
    if len(positionsList4) < len(positionsListNot4):
        # There are more 0's than 4's.
        random.shuffle(positionsListNot4)
        positionsListNot4 = positionsListNot4[:len(positionsList4)]
    else:
        # There are more 4's than 0's
        random.shuffle(positionsList4)
        positionsList4 = positionsList4[:len(positionsListNot4)]

    outputTargetsList = [targetsList[i] for i in positionsList4]
    outputTargetsList += [targetsList[i] for i in positionsListNot4]

    # The result are save in output lists
    featuresOutputList = [featuresList[i] for i in positionsList4]
    featuresOutputList += [featuresList[i] for i in positionsListNot4]

    return featuresOutputList, outputTargetsList


def activateDeativateFeatures(featuresListFull, DICT_FEATURES_POSITIONS):
    """
    Given a list containing all the features, it is evaluated whether each features is kept or discarded
    """
    activeFeaturesOutputList = None

    # iterate over the dict which indicate whether each position is ON/OFF
    decisionsList = [] # It's going to be a list of True/False values
    for oneValue in DICT_FEATURES_POSITIONS.items():
        auxIndex = (oneValue[0].split('-'))[0]
        decision = oneValue[1]
        decisionsList += [decision]

    positionListToDelete = [i for i, e in enumerate(decisionsList) if e == False]

    # from sublists are deleted those position according to False values in the dict
    activeFeaturesOutputList = [[y for i,y in enumerate(x) if i not in positionListToDelete]for x in featuresListFull]
    return activeFeaturesOutputList


def getFeaturesAndTargetsAccordingToTargetValue(featuresList, targetsList, targetValue, targetCondition):
    """
    Given two lists with targets and values whose targets are 'targetValue'
    :param targetCondition If true: are chosen features and targets according to certain value
                           If false: are chosen features and targets are not according to certain value.
    """
    positionListWithTargetValue = []
    if targetCondition:
        positionListWithTargetValue = [i for i, e in enumerate(targetsList) if e == targetValue]
    else:
        positionListWithTargetValue = [i for i, e in enumerate(targetsList) if e != targetValue]
    return [featuresList[index] for index in positionListWithTargetValue], [targetsList[index] for index in positionListWithTargetValue]


def concatListIntoString(inputList, separator):
    """
    Given a list, return a string with all the list values with the separator between each pair of values
    """
    outputString = ""
    # se evalua si la inputList de verdad lo es
    if isinstance(inputList, collections.Iterable):
        for anElement in inputList:
            outputString += separator + str(anElement)
        outputString = outputString[len(separator):]
    else:
        outputString = str(inputList)
    return  outputString


def replaceValuesInList(inputList, searchValue, newValue, findEqual):
    """
    Given a list, value to replace and a boolean value (findEqual), the replacement is with the new value
    """
    outputList = []
    if findEqual:
        outputList = [x if x != searchValue else newValue for x in inputList]
    else:
        outputList = [x if x == searchValue else newValue for x in inputList]
    return outputList


def convertKeysValuesDictIntoString(inputDict, keysSeparator, valuesSeparator):
    keysString = ""
    valuesString = ""
    for oneKey in inputDict:
        keysString += str(keysSeparator) + str(oneKey)
        valuesString += str(valuesSeparator) + str(inputDict[oneKey])
    #  Separators at the beginning of the string are discarded.
    keysString = keysString[len(keysSeparator):]
    valuesString = valuesString[len(valuesSeparator):]
    return keysString, valuesString


def getAvgElementsSublists(inputList, returnString):
    """
    Given a list counts elements of each sublist and calculate the amount average.
    :param returnString: if True, the answer will be a string. Otherwise, returns two numeric values (average and elements' amount)
    """
    elementsAmount = 0
    for anElement in inputList:
        elementsAmount += len(anElement)
    average = elementsAmount/len(inputList)
    if int(average) == average:
        average = int(average)

    if returnString:
        avgStringAndElemAmount = str(average) + "(" + str(elementsAmount) + "/" + str(len(inputList)) + ")"
        return avgStringAndElemAmount
    else:
        return average, elementsAmount


def calculateTPTNFPFN(predictionList, targetsList, labels, returnString):
    """
    Calculates  TP, TN, FP, FN values.
    :param returnString: If True, the answer will be a string. Otherwise, just return numeric values
    """
    # params: trueTargets, predictionsTargets
    [tn, fp, fn, tp] = confusion_matrix(predictionList, targetsList, labels=labels).ravel()

    if returnString:
        stringTN = str(tn) + "(" + str(round(tn / len(predictionList), 2)) + "%)"
        stringFP = str(fp) + "(" + str(round(fp / len(predictionList), 2)) + "%)"
        stringFN = str(fn) + "(" + str(round(fn / len(predictionList), 2)) + "%)"
        stringTP = str(tp) + "(" + str(round(tp / len(predictionList), 2)) + "%)"
        return stringTN, stringFP, stringFN, stringTP
    else:
        return tn, fp, fn, tp


def calculateAccPrecRecF1(targetsList, predictionsList, convertTo1sAndMinus1, inlierValue):
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    if convertTo1sAndMinus1:
        #  Before appluing the metrics, it´s needed convert targetsList to 1s and -1s
        # 1: inliers, -1: outliers
        #  train_target, first 4 values are replace with 1, and then 0 are replace with -1.
        targetsList = replaceValuesInList(targetsList, valueToSearch=inlierValue,
                                                newValue=1, lookForEqual=True)  # Replace inlierValue (4's) by 1 because they are inliers
        targetsList = replaceValuesInList(targetsList, valueToSearch=1,
                                                newValue=-1, lookForEqual=False)  # Replace (not 4's) by -1 because they are outliers

    accuracy = metrics.accuracy_score(targetsList, predictionsList)
    precision = metrics.precision_score(targetsList, predictionsList)
    recall = metrics.recall_score(targetsList, predictionsList)
    f1 = metrics.f1_score(targetsList, predictionsList)

    return accuracy, precision, recall, f1


def replaceValuesInList(inputList, valueToSearch, newValue, lookForEqual):
    """
    Given a list replace valueToSearch with newValue
    :param lookForEqual If True, compare by =. Otherwise, compare by distinct
    """
    answerList = []
    if lookForEqual:
        answerList = [x if x != valueToSearch else newValue for x in inputList]
    else:
        answerList = [x if x == valueToSearch else newValue for x in inputList]
    return answerList


def mixupKernelsAndGammas(kernel, gammasList):
    """
    Returns something like this
    [{'kernel': 'rbf', 'gamma': 0.1}, {'kernel': 'rbf', 'gamma': 0.2}, {'kernel': 'rbf', 'gamma': 0.3}]
    """
    outputList = []
    for oneGammaValue in gammasList:
        outputList += [{'kernel':str(kernel), 'gamma':oneGammaValue}]
    return outputList


def mixUpLists(inputList1, list1, inputList2, list2):
    outputList = []
    for oneValue1 in list1:
        if not(isinstance(oneValue1, bool)):
            oneValue1 = str(oneValue1)
        for oneValue2 in list2:
            if not(isinstance(oneValue2, bool)):
                oneValue2 = str(oneValue2)
            outputList += [{inputList1:oneValue1, inputList2:oneValue2}]
    return outputList


def generateFilesForEachStringInList(inputList, folderPath, interval, kValue, headerString, OCEANtype=None):
    """
    Generates a dict of open files for the classifiers from the input list
    """
    OCEANstring = ""
    if OCEANtype == None:
        OCEANstring = constants.CADENA_TIPO_OCEAN
    else:
        OCEANstring = OCEANtype

    dictFromFiles = {}
    for oneClassifier in inputList:
        auxFileNameWithPath = folderPath+"config-" + OCEANstring +"-"+str(interval)+",k"+str(kValue)+"-"+str(oneClassifier)+".txt"
        dictFromFiles[str(oneClassifier)] = open(auxFileNameWithPath, "w")
        # check whether the file is empty to add the headers
        if os.path.getsize(auxFileNameWithPath) == 0:
            dictFromFiles[str(oneClassifier)].write(headerString)
    return dictFromFiles


def getOCEANarray(userID, OCEANfile):
    """
    Given a user ID and a file name, returns its OCEAN value.
    If the user does not exist return None
    """
    userID = str(userID)
    answer = None
    OCENAinputFile = open("preprocessed files/"+OCEANfile, "r", encoding='UTF8')

    isFirstLine = True
    for oneLine in OCENAinputFile:
        # The first line is discarded because is the columns' header
	    if not (isFirstLine):
            arrayLineFileOCEAN = oneLine.split('\t')
            if arrayLineFileOCEAN[0] == userID:
                answer = int(arrayLineFileOCEAN[1].strip())
                if answer == -1:
                    answer = None
                break
        else:
            isFirstLine = False
    return answer


def replaceValueInEachSublistGivenPosition(inputList, position, newValue):
    """
    Given a list, a position(which starts in 0) and a value,
    replace in every sublist by the new value in certain position
    """
    answerList = []
    for oneSublist in inputList:
        oneSublist[position] = newValue
        answerList +=  [oneSublist]
    return answerList