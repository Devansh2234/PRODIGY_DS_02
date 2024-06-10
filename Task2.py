import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import sklearn
import os
import re


df = pd.read_csv("energy_dataset_.csv")
df.head()

missing_values = df.isnull().sum()
print(missing_values)
sns.set(style="whitegrid")

# Installed Capacity of Different Types of Renewable Energy
plt.figure(figsize=(10, 6))
sns.barplot(x='Type_of_Renewable_Energy', y='Installed_Capacity_MW', data=df, ci=None, palette='viridis')
plt.title('Installed Capacity of Different Types of Renewable Energy')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Installed Capacity (MW)')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()
