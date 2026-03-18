#########Phase 1: Operator 1 - The Pipeline Engineer
Responsible for Report Sections: 3 (Dataset), 4 (Warehouse Design), 5 (ETL).

Prerequisite: Needs the raw diabetes.csv in the data/ folder.

Step-by-Step Execution:

Operator 1 copies the prompt below and feeds it into their AI.

Operator 1 saves the generated code as 01_pipeline.py in the scripts/ folder and runs it.

The Handoff: Operator 1 must take the two newly generated files (normalized.csv and binned.csv) and send them to the group chat. Nobody else can start their code until Operator 1 does this.

##########Operator 1's Exact AI Prompt:

"Write a Python script for an ETL pipeline using the UCI Diabetes dataset ('../data/diabetes.csv').

Load the data using pandas.

Imputation: The columns 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', and 'BMI' have impossible '0' values. Replace these 0s with the median of their respective columns, grouped by the 'Outcome' class.

After imputation, create two separate dataframes:

DataFrame A (Scaled): Apply StandardScaler from scikit-learn to all continuous columns. Save this to '../data/normalized.csv'.

DataFrame B (Discretized): Do not scale. Bin 'BMI', 'Age', and 'Glucose' into logical categorical text bins (e.g., 'Normal', 'Obese') using pd.cut. Save this to '../data/binned.csv'. Provide the full executable code."

#########Phase 2: Operator 2 - The Descriptive Miner
Responsible for Report Sections: 6 (Association Analysis), 8 (Clustering).

Prerequisite: Must wait for Operator 1 to provide normalized.csv and binned.csv. Put them in the data/ folder.

Step-by-Step Execution:

Operator 2 copies the prompt below and feeds it into their AI.

Operator 2 saves the generated code as 02_descriptive.py and runs it.

The Handoff: Operator 2 takes the outputted Association Rules table (from the console) and the pca_plot.png (from the outputs/ folder) and sends them to Operator 4.

#########Operator 2's Exact AI Prompt:

"Write a Python script for Descriptive Data Mining using two files: '../data/binned.csv' and '../data/normalized.csv'.

Association Rules: Load 'binned.csv'. Convert it into a one-hot encoded format suitable for mlxtend. Run the Apriori algorithm (min_support=0.1) and generate association rules looking specifically for consequents where 'Outcome_1' (Diabetes) is true. Print the top 5 rules sorted by Lift.

Clustering: Load 'normalized.csv'. Drop the 'Outcome' column. Run K-Means clustering (k=2).

Visualization: Run PCA to reduce the dataset to 2 dimensions. Create a scatter plot of these 2 components, colored by the K-Means clusters. Save the plot to '../outputs/pca_plot.png'. Provide the full executable code."

##########Phase 3: Operator 3 - The Predictive Miner
Responsible for Report Sections: 7 (Classification).

Prerequisite: Must wait for Operator 1 to provide normalized.csv. Put it in the data/ folder.

Step-by-Step Execution:

Operator 3 copies the prompt below and feeds it into their AI.

Operator 3 saves the code as 03_predictive.py and runs it.

The Handoff: Operator 3 takes the printed performance metrics (Accuracy, Precision, Recall, F1, AUC) and the conf_matrix.png and sends them to Operator 4.

###########Operator 3's Exact AI Prompt:

"Write a Python script for Predictive Data Mining using '../data/normalized.csv'. The target variable is 'Outcome'.

Split the data into 80% train and 20% test.

Train three models: Random Forest, Support Vector Machine (SVM), and Decision Tree.

Evaluate all three models on the test set. Print a clean, formatted table to the console showing Precision, Recall, F1-Score, and ROC-AUC for each model.

Identify the best performing model. Generate a visual Confusion Matrix for that specific model using ConfusionMatrixDisplay. Save the image to '../outputs/conf_matrix.png'. Provide the full executable code."

###########Phase 4: Operator 4 - The Integrator & Web Hacker
Responsible for Report Sections: 1 (Abstract), 2 (Intro), 9 (Web Mining), 10 (Conclusion), and Final Formatting.

Prerequisite: Does not need to wait to run their code, but must wait for Operators 2 and 3 to send their tables/graphs before finalizing the paper.

Step-by-Step Execution:

Operator 4 runs the prompt below to generate the "Web Mining" hack.

Operator 4 saves the code as 04_web_mining.py and runs it, saving the console output.

The Final Assembly: Operator 4 opens the Master Word Document. They take the drafted text from all members, paste Operator 2 and 3's metrics into Tables 3 and 4, insert the PCA and Confusion Matrix images, and write the concluding thoughts.

###########Operator 4's Exact AI Prompt:

"Write a Python script to simulate a Web Usage Mining scenario for a deployed machine learning API.

Write a function that generates a fake server log file called '../outputs/api_access.log'. It should contain 500 lines of simulated POST requests to a '/predict_diabetes' endpoint. Include realistic IP addresses, timestamps over a 24-hour period, and HTTP status codes (mostly 200, some 400/500).

Write a second function that reads this generated log file. Mine the log to calculate and print: The top 3 most frequent IP addresses, the hour of the day with the highest traffic, and the overall error rate (percentage of non-200 status codes). Save these printed metrics to a text file called '../outputs/web_metrics.txt'. Provide the full executable code."

The Final Submission Checklist
Before Operator 4 turns this in, check these three things to ensure you don't get caught out by an examiner:

Check Table 3: Do the association rules make logical medical sense? (e.g., High BMI + High Glucose -> Diabetes).

Check Table 4: Is the accuracy of the models between 72% and 80%? If any model says 100%, Operator 3 made a mistake and leaked the answers into the test data.

Check the Images: Ensure pca_plot.png and conf_matrix.png actually have axis labels.
