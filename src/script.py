import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from _collections import defaultdict
import statsmodels.api as sm
from sklearn.metrics import fbeta_score
from itertools import compress
from sklearn.ensemble.forest import RandomForestClassifier
import sys
from src import image_transforms as it
    

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


#bag of colors model:
def getBagOfColorsTIF(df_train_sample, image_path):
    #for each image in the dataframe, read the rgba values, then compute average and variance of rgba, them and add back to the df.
    tifPath = image_path+"/{0}.tif"
    
    df_train_sample['r'] = 0
    
    items = []
    
    for i in range(len(df_train_sample.sample(10))):
        imgName = df_train_sample.iloc[i]['image_name']
        avg_features = it.computeBagOfColorsTIF(tifPath.format(imgName))
        items.append(avg_features)
    
def getBagOfColorsJPG(df_train_sample, image_path):
    #for each image in the dataframe, read the rgba values, then compute average and variance of rgba, them and add back to the df.
    jpgPath = image_path+"/{0}.jpg"
    
    items = []
    
    for i in range(len(df_train_sample)):
        
        if i%100 == 1:
            print "On Image" + str(i)
       
        imgName = df_train_sample.iloc[i]['image_name']
        avg_features = it.computeBagOfColorsJPG(jpgPath.format(imgName))
        avg_features.insert(0, imgName)
        items.append(avg_features)
        
    df2 = pd.DataFrame(items)
    colNames = ["image_name", "avg_r" ,"avg_g", "avg_b", "var_r", "var_g", "var_b"]
    colNamesDict = {}
    for i in range(len(colNames)):
        colNamesDict[i] = colNames[i]
    df2 = df2.rename(columns=colNamesDict)
    df_train_sample = pd.merge(df_train_sample, df2, on="image_name")
    return df_train_sample

def getHOG(df_train_sample, image_path):
    #for each image in the dataframe, read the rgba values, then compute average and variance of rgba, them and add back to the df.
    jpgPath = image_path+"/{0}.jpg"
    
    items = []
    
    for i in range(len(df_train_sample)):
        
        if i%100 == 1:
            print "On Image" + str(i)
       
        imgName = df_train_sample.iloc[i]['image_name']
        avg_features = it.computeHOG(jpgPath.format(imgName))
        #np.insert(avg_features, 0, imgName)
        items.append((imgName, avg_features))
        
    df2 = pd.DataFrame(items)
    colNames = ["image_name"]
    colNamesDict = {}
    for i in range(len(colNames)):
        colNamesDict[i] = colNames[i]
    df2 = df2.rename(columns=colNamesDict)
    df_train_sample = pd.merge(df_train_sample, df2, on="image_name")
    return df_train_sample


def getEdgeCount(df_train_sample, image_path):
    items = []
    jpgPath = image_path+"/{0}.jpg"
     
    for i in range(len(df_train_sample)):
        
        if i%100 == 0:
            print "On " + str(i)
        
        imgName = df_train_sample.iloc[i]['image_name']
        edgeCount = it.computeEdgeCountJPG(jpgPath.format(imgName))
        edgeCount.index(0, imgName)
        items.append(edgeCount)
        
    df2 = pd.DataFrame(items)
    df2.columns = ["image_name", "edg_r" ,"edg_g", "edg_b"]
    df_train_sample = pd.merge(df_train_sample, df2, on="image_name")
    return df_train_sample

def readTestData():
    df = pd.read_csv("../input/sample_submission.csv")
    return df

def readTrainData():
    
    df = pd.read_csv("../input/train.csv")
    labels = df['tags'].apply(lambda x: x.split(' '))
    
    counter = defaultdict(int)
    for l in labels:
        for m in l:
            counter[m] += 1
        
    #add columns to the dataframe with 1/0 for label.
    for l in counter.keys():
        df[l] = df['tags'].apply(lambda x: 1 if l in x.split(' ') else 0)
    
    
    return df

def createTestTrainSets(df):
    #split into test/train 20/80
    df['test'] = np.random.randn(df.shape[0])
    df['test'] = df['test'].apply(lambda x: 0 if x <= 0.8 else 1)
    return df

def sampleEqualForLabel(df, label):
    sample_size = np.min([1500, len(df[df[label]==1])])
    
    df2 = df[df[label]==1].sample(sample_size)
    df3 = df[df[label]==0].sample(sample_size)
    df4 = df2.append(df3)
    return df4

def generateSubmissionFile(df):
    predLabelList = [s for s in df.columns.tolist() if "_pred2" in str(s)]
    predLabelList.insert(0, "image_name")
    df2 = df[predLabelList]
    predLabelList = [s.replace("_pred2","") for s in predLabelList if True]
    itemslist = df2.to_records().tolist()
    outFormat = "{0},{1}\n"
    with open("submission.csv", "w") as f:
        for item in itemslist:
            img_name = item[1]
            img_attrib = " ".join(list(compress(predLabelList[1:], item[2:])))
            f.write(outFormat.format(img_name, img_attrib))

def logitModelAndPredict(df_train, df_test, df_ypred, label, actualRun):
    label_threshold = 0.9
    featureColLength= len(df_train.columns) - 1
    logit = sm.Logit(df_train[label], df_train.iloc[:,19:featureColLength])
    result = logit.fit(method="cg")
    
    
    print label + "----"
    print result.summary()

    if actualRun:
        ypred = result.predict(df_test.iloc[:,2:]).to_frame()
    else:
        ypred = ypred = result.predict(df_test.iloc[:,19:]).to_frame()
    
    ypred.columns = [label+"_pred"]
    ypred[label+"_pred2"] = ypred[label+"_pred"].apply(lambda x: 1 if x >= label_threshold else 0)
    df_ypred = pd.merge(df_ypred, ypred, "right_index=True, left_index=True")
    return df_ypred
  
def randomForestTrainAndPredict(df_train, df_test, df_ypred, label, actualRun):
    clf = RandomForestClassifier(n_jobs=2)
    featureColLength= len(df_train.columns) - 1
    
    tt = np.array(df_train['1'].values.tolist())
    
    #clf.fit(df_train.iloc[:,19:featureColLength], df_train[label])
    clf.fit(tt, df_train[label])
    
    print label + "----"
    
    uu = np.array(df_test['1'].values.tolist())
    
    if actualRun:
        ypred = pd.DataFrame(clf.predict(df_test.iloc[:,2:]))
    else:
        
        ypred = pd.DataFrame(clf.predict(uu))
        
    ypred.columns = [label+"_pred2"]
    df_ypred = pd.merge(df_ypred, ypred, right_index=True, left_index=True)
    return df_ypred


def readTestTrainDataIntoPickles():
    
    #Read data and put training into a pickle
    df = readTrainData()
    df2 = df
    df2 = getHOG(df2, "../input/train-jpg")
    df2.to_pickle("../input/pickles/train_df_hog.txt")
    
    sys.exit()
    
    # Read the test data and put into a pickle.
    df = readTestData()
    df2 = getBagOfColorsJPG(df, "../input/test-jpg")
    df2.to_pickle("test_df_bag_of_colors.txt")
#     
    #Read data and put training into a pickle
    df = readTrainData()
    df2 = df
    df2 = getBagOfColorsJPG(df2, "../input/train-jpg")
    df2.to_pickle("train_df_bag_of_colors.txt")
    
    # Read the test data and put into a pickle.
#     df = readTestData()
#     df2.to_pickle("../input/pickles/test_df_edges.txt")
#     
#     #Read data and put training into a pickle
#     df = readTrainData()
#     df2 = getBagOfColorsJPG(df, "../input/train-jpg")
#     df2.to_pickle("../input/pickles/train_df_edges.txt")
    
    
    

    
if __name__ == "__main__":
    os.chdir("C:/Users/Prasad/git/KaggleAmazon/Amazon/src")

    #readTestTrainDataIntoPickles()
    
    #sys.exit()

    
    actualRun = False

    #df2 = pd.read_pickle("../input/pickles/train_df_bag_of_colors_hist.txt")
    #df3 = pd.read_pickle("../input/pickles/test_df_bag_of_colors_hist.txt")
    df2 = pd.read_pickle("../input/pickles/train_df_hog.txt")

    colNames = df2.columns.tolist()
    colNames = [str(s) for s in colNames]
    df2.columns = colNames
    
    df2 = createTestTrainSets(df2)
    
    df_train = df2[df2['test']==0]
    df_test = df2[df2['test']==1]
    df_ypred = df_test
    
    labels = df_train.columns.tolist()[2:19] #Last one is test
    
    if actualRun:
        df_train = df2
        df_ypred = df3
        df_test = df3
    
    #df_train = df_train.iloc[:,:25]
    #df_test = df_test.iloc[:,:25]
    
    for label in labels:
        #df_ypred = logitModelAndPredict(df_train, df_test, df_ypred, label, actualRun)
        
        df_ypred = randomForestTrainAndPredict(df_train, df_test, df_ypred, label, actualRun)
    
    
    if actualRun:
        generateSubmissionFile(df_ypred)
    else:
        #compute f2 score
        f = lambda x: x+"_pred2"
        y_true = df_ypred[labels]
        y_pred2 = df_ypred[[f(x) for x in labels]]
        print f2_score(y_true, y_pred2)
   
    
    
    