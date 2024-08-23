import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
np.random.seed(40)

folder_path = 'SampleSubmission'
os.makedirs(folder_path, exist_ok=True)

df_train = pd.read_csv('train_gen1.csv')
X_test = pd.read_csv('public_X_test.csv')
df_variants = pd.read_csv('variants.csv')

print('Subsampling the training data to speed up the training process ...')
df_train = df_train.sample(frac=0.2)

print('df_train.shape:', df_train.shape)
print('X_test.shape:', X_test.shape)
print('df_variants.shape:', df_variants.shape)

# impute missing values
df_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

classifier = DecisionTreeClassifier(random_state=42)

X_train = df_train.drop(columns=['Timesteps', 'ChassisId_encoded', 'gen', 'risk_level'])
y_train = df_train['risk_level']

print('fitting the classifier ...')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test.drop(columns=['Timesteps', 'ChassisId_encoded', 'gen']))
df_pred = pd.DataFrame(data=y_pred, columns=['pred'])
df_pred.to_csv('SampleSubmission/prediction.csv', index=False)

# evaluating the prediction agianst a mocked up ground truth 
length = len(X_test)
sequence = np.array(['Low', 'Medium', 'High'])
mocked_true = sequence[np.arange(length) % len(sequence)]

y_true = pd.DataFrame()
y_true['risk_level'] = mocked_true
y_true['pred'] = y_pred
y_true['gen'] = X_test['gen']

def macro_avg_f1(y_true, y_pred):
    # Get precision, recall, and f1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)

    # Calculate macro average F1-score
    macro_avg_f1 = sum(f1) / len(f1)

    return round(macro_avg_f1, 2)

print('*** Checking Performance of the model agianst mocked up truth labels ***')
score1 = macro_avg_f1(y_true[y_true['gen'] == 'gen1']['risk_level'],
                            y_true[y_true['gen'] == 'gen1']['pred'])
score2 = macro_avg_f1(y_true[y_true['gen'] == 'gen2']['risk_level'],
                            y_true[y_true['gen'] == 'gen2']['pred'])

scores = {
    'Final Score': round((score1 + score2)/2, 2),
    'score on gen1:': score1,
    'score on gen2:': score2
}

print(f'Evaluation results against mocked up ground truth: {macro_avg_f1(mocked_true, y_pred)}')
print('\nScores Details:\n', scores)

def compress_file(input_file, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file, os.path.basename(input_file))

input_file = 'SampleSubmission/prediction.csv'  # Input file to compress
output_zip = 'SampleSubmission/prediction.csv.zip'  # Output ZIP archive
compress_file(input_file, output_zip)
print(f'File "{output_zip}" successfully created for submission to the Codabench portal.')
