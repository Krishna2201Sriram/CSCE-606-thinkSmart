def openTrainedModel(modelPath="../models/dummy.txt"):
    try:
        f = open(modelPath,"rb")
        return f
    # Do something with the file
    except IOError:
        print("The {} not accessible".format(modelPath));
        return None
    

def retFileName(pathToFile="../data/dummy.csv"):
    parsed = pathToFile.split('/');
    parsed = parsed[-1].split('.');
    return parsed[0];


'''
Input Parameters: path_to_csv (to be trained or predicted),subject_name,grade
Output: 1(predicts the scores for input csv)
        2(trains the model for input csv)

Raise:
        -> if there is no file in the given datapath(done)
        -> if the given input_csv has no y_value and trained model(TO BE IMPLEMENTED)
'''
def learn(path_to_csv="../data/dummy.csv",subject_name=None,grade=None):
    #This is the y_column that is slicing from the input csv for training and dropped during prediction
    #Needs to be changed depending on the input csv y_column
    responseVariable = 'Spring 2019 STAAR\nMA05\nPcntScore\n5/2019 or 6/2019'; 
    
    if subject_name is None or grade is None:
        raise("Please provide proper subject_name and grade of the excel sheet");
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import pickle

    #Model Naming Convention: filename_pickle
    #Check if a trained model exist for this dataSheet
    
    trainedModel = None;


    openedModel = openTrainedModel("../models/"+retFileName(path_to_csv)+"_pickle");
    if openedModel is None:
        print("No trained model available for this grade and subject ... to be trained");
    else:
        trainedModel = pickle.load(openedModel);
        openedModel.close();
        
        

    if trainedModel is not None:
        #Return the infered result as csv.
        trainedCSV = pd.read_csv(path_to_csv);
        
        originalCSV = trainedCSV;

        trainedCSV=trainedCSV.fillna(trainedCSV.mean())

        trainedCSV = pd.get_dummies(trainedCSV,columns=['Ethnicity']);
        if responseVariable in trainedCSV.columns:
            trainedCSV = trainedCSV.drop(responseVariable,axis=1);
        
        trainedCSV = trainedCSV.iloc[:,2:];  
        
        predictedVals = trainedModel.predict(trainedCSV);
        
        #tempCSV = pd.DataFrame({'Original': originalCSV, 'Predicted': predictedVals})
        originalCSV['predictedScores'] = predictedVals;
        
        originalCSV.to_csv("../predicted_data/"+"predicted_"+retFileName(path_to_csv)+".csv", index = False)
        return 1;
        

    math = pd.read_csv(path_to_csv);#not required to close this file
    
    math=math.fillna(math.mean())
    
    math = pd.get_dummies(math,columns=['Ethnicity'])

    math_analysis=math.iloc[:,2:]
    x_math=math_analysis.drop(responseVariable,axis=1)
    y_math=math_analysis[responseVariable]
    
    X_train_math, X_test_math, y_train_math, y_test_math = train_test_split(x_math, y_math, test_size=0.2, random_state=42);
    regressor = LinearRegression();
    print("The model is being trained ...");
    regressor.fit(X_train_math, y_train_math);

    regressor.score(X_test_math,y_test_math)
    
    #
    pickle.dump(regressor,open("../models/"+retFileName(path_to_csv)+"_pickle",'wb'));
    print("The trained model has been saved to path: {}".format("../models/"+retFileName(path_to_csv)+"_pickle"));
    
    return 2;


if __name__ == "__main__":
    learn("../data/math_7th.csv","math",7);