import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


exp_data = pd.read_csv('generated/experiments.csv', names=['labels_per_class', 'verb_type', 'representation_type', 'accuracy'])
captions = exp_data[exp_data['representation_type'] == 'e_caption']
object_labels = exp_data[exp_data['representation_type'] == 'e_object']
full_text = exp_data[exp_data['representation_type'] == 'e_combined']

cap_x_data = captions['labels_per_class'].unique()
cap_y_data = captions.groupby('labels_per_class').mean().to_numpy()
cap_y_err = captions.groupby('labels_per_class').std().to_numpy()

obj_x_data = object_labels['labels_per_class'].unique()
obj_y_data = object_labels.groupby('labels_per_class').mean().to_numpy()
obj_y_err = object_labels.groupby('labels_per_class').std().to_numpy()

text_x_data = full_text['labels_per_class'].unique()
text_y_data = full_text.groupby('labels_per_class').mean().to_numpy()
text_y_err = full_text.groupby('labels_per_class').std().to_numpy()


plt.style.use('seaborn-notebook')

fig, ax = plt.subplots()
ax.errorbar(cap_x_data, cap_y_data, yerr=cap_y_err)
ax.errorbar(obj_x_data, obj_y_data, yerr=obj_y_err)
ax.errorbar(text_x_data, text_y_data, yerr=text_y_err)
ax.legend(['Captions', 'Objects annotations', 'Captions+Objects'])

ax.set_title('Semi-supervised GTG (Text data)')
ax.set_xlabel('#labels')
ax.set_ylabel('Accuracy')
plt.show()

