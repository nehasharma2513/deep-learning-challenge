# deep-learning-challenge

In this project we are helping the nonprofit foundation Alphabet Soup to build a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using the features in the provided dataset we are creating a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, we have a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

**EIN and NAME**—Identification columns
**APPLICATION_TYPE**—Alphabet Soup application type
**AFFILIATION**—Affiliated sector of industry
**CLASSIFICATION**—Government organization classification
**USE_CASE**—Use case for funding
**ORGANIZATION**—Organization type
**STATUS**—Active status
**INCOME_AMT**—Income classification
**SPECIAL_CONSIDERATIONS**—Special consideration for application
**ASK_AMT**—Funding amount requested
**IS_SUCCESSFUL**—Was the money used effectively

**Step 1: Preprocess the Data**

By using the knowledge of Pandas and scikit-learn’s StandardScaler(), we need to preprocess the dataset. This step prepares us for Step 2, where we'll compile, train, and evaluate the neural network model.

1. Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
2. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
3. What variable(s) are the target(s) for your model?
   - IS_SUCCESSFUL is the target variable
4. What variable(s) are the feature(s) for your model?

   - APPLICATION_TYPE
   - AFFILIATION
   - CLASSIFICATION
   - USE_CASE
   - ORGANIZATION
   - STATUS
   - INCOME_AMT
   - SPECIAL_CONSIDERATIONS
   - ASK_AMT

5. Drop the EIN and NAME columns.

6. Determine the number of unique values for each column.

7. For columns that have more than 10 unique values, determine the number of data points for each unique value.

8. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

9. Use pd.get_dummies() to encode categorical variables.

10. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

11. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

**Step 2: Compile, Train, and Evaluate the Model**

Using the knowledge of TensorFlow, we'll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. We’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

**Step 3: Optimize the Model**

Using the knowledge of TensorFlow, optimize the model to achieve a target predictive accuracy higher than 75%.

Using any or all of the following methods to optimize your model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
- Dropping more or fewer columns.
- Creating more bins for rare occurrences in columns.
- Increasing or decreasing the number of values for each bin.
- Add more neurons to a hidden layer.
- Add more hidden layers.
- Use different activation functions for the hidden layers.
- Add or reduce the number of epochs to the training regimen.
