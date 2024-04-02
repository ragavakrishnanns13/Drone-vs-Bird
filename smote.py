import pandas as pd
from imblearn.over_sampling import SMOTE

def resampling(filename):

    data = pd.read_csv(str(filename))

    X = data.drop('target', axis=1)  
    y = data['target']

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_data['target'] = y_resampled  

    resampled_data.to_csv('resampled.csv', index=False)

    print("SMOTE applied successfully and resampled data saved to 'resampled_output_file.csv'.")
