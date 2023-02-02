import matplotlib.pyplot as plt
import pandas as pd
import os


root_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.abspath(os.path.join(root_dir, 'input', 'train'))
test_dir = os.path.abspath(os.path.join(root_dir, 'input', 'test'))

train_df = pd.read_csv(os.path.join(root_dir, 'input', 'train_labels.csv'))
label_counts = train_df['label'].value_counts()*100/len(train_df)
labels = ['No cancer', 'Cancer']

fig1, ax1 = plt.subplots()
ax1.pie(label_counts, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')
plt.show()
