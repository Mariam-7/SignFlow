import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Define the folder path
folder_path = '/Users/mariamhafeez/Desktop/Hackie/sign-language-detector-python'

# Load the data from the pickle file located in the specified folder
data_dict = pickle.load(open(os.path.join(folder_path, 'data.pickle'), 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model in the same folder
with open(os.path.join(folder_path, 'model.p'), 'wb') as f:
    pickle.dump({'model': model}, f)
