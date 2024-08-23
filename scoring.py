import json
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import pandas as pd

def macro_avg_f1(y_true, y_pred):
    # Get precision, recall, and f1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)

    # Calculate macro average F1-score
    macro_avg_f1 = sum(f1) / len(f1)

    return round(macro_avg_f1, 3)


reference_dir = os.path.join('/app/input', 'ref')
prediction_dir = os.path.join('/app/input', 'res')
score_dir = '/app/output'
print('pd.__version__:', pd.__version__)

print('Reading prediction')
prediction = pd.read_csv(os.path.join(prediction_dir, 'prediction.csv'))
y_true = pd.read_csv(os.path.join(reference_dir, 'private_y_test'))

y_true['pred'] = prediction['pred']

print('*** Checking Performance on the whole data ***')
score1 = macro_avg_f1(y_true[y_true['gen'] == 'gen1']['risk_level'],
                            y_true[y_true['gen'] == 'gen1']['pred'])
score2 = macro_avg_f1(y_true[y_true['gen'] == 'gen2']['risk_level'],
                            y_true[y_true['gen'] == 'gen2']['pred'])
print('Scores:')
scores = {
    'avg': round((score1 + score2)/2, 3) ,
    'score1': score1,
    'score2': score2
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))

