import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import sklearn
import os
import re

# Load the dataset
df = pd.read_csv("/Users/dhartipatel/Documents/GitHub/PRODIGY_DS_02/energy_dataset_.csv")

# Display missing values
missing_values = df.isnull().sum()
print(missing_values)

sns.set(style="whitegrid")

# Installed Capacity of Different Types of Renewable Energy
plt.figure(figsize=(10, 6))
sns.barplot(x='Type_of_Renewable_Energy', y='Installed_Capacity_MW', data=df, ci=None, palette='coolwarm')
plt.title('Installed Capacity of Different Types of Renewable Energy')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Installed Capacity (MW)')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()

# Energy Production vs. Consumption
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Energy_Production_MWh', y='Energy_Consumption_MWh', hue='Type_of_Renewable_Energy', palette='coolwarm', data=df)
plt.title('Energy Production vs. Consumption')
plt.xlabel('Energy Production (MWh)')
plt.ylabel('Energy Consumption (MWh)')
plt.legend(title='Type of Renewable Energy', labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()

# Storage Efficiency of Different Types of Renewable Energy
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type_of_Renewable_Energy', y='Storage_Efficiency_Percentage', data=df, palette='coolwarm')
plt.title('Storage Efficiency of Different Types of Renewable Energy')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Storage Efficiency (%)')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()

# Initial Investment vs. GHG Emission Reduction
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Initial_Investment_USD', y='GHG_Emission_Reduction_tCO2e', hue='Type_of_Renewable_Energy', palette='coolwarm', data=df)
plt.title('Initial Investment vs. GHG Emission Reduction')
plt.xlabel('Initial Investment (USD)')
plt.ylabel('GHG Emission Reduction (tCO2e)')
plt.legend(title='Type of Renewable Energy', labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()

# Number of Jobs Created by Different Types of Renewable Energy
plt.figure(figsize=(10, 6))
sns.barplot(x='Type_of_Renewable_Energy', y='Jobs_Created', data=df, ci=None, palette='coolwarm')
plt.title('Number of Jobs Created by Different Types of Renewable Energy')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Number of Jobs Created')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Detailed Breakdown by Energy Type
grouped_df = df.groupby('Type_of_Renewable_Energy').mean().reset_index()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.barplot(x='Type_of_Renewable_Energy', y='Installed_Capacity_MW', data=grouped_df, palette='coolwarm')
plt.title('Installed Capacity by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Installed Capacity (MW)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.subplot(1, 3, 2)
sns.barplot(x='Type_of_Renewable_Energy', y='Energy_Production_MWh', data=grouped_df, palette='coolwarm')
plt.title('Energy Production by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Energy Production (MWh)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.subplot(1, 3, 3)
sns.barplot(x='Type_of_Renewable_Energy', y='Energy_Consumption_MWh', data=grouped_df, palette='coolwarm')
plt.title('Energy Consumption by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Energy Consumption (MWh)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.tight_layout()
plt.show()

# Efficiency Improvement Opportunities
efficiency_summary = df.groupby('Type_of_Renewable_Energy')['Storage_Efficiency_Percentage'].describe()
print(efficiency_summary)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Type_of_Renewable_Energy', y='Storage_Efficiency_Percentage', data=df, palette='coolwarm')
plt.title('Storage Efficiency by Type of Renewable Energy')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Storage Efficiency (%)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()

# Financial Metrics
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Type_of_Renewable_Energy', y='Initial_Investment_USD', data=grouped_df, palette='coolwarm')
plt.title('Initial Investment by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Initial Investment (USD)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.subplot(1, 2, 2)
sns.barplot(x='Type_of_Renewable_Energy', y='Financial_Incentives_USD', data=grouped_df, palette='coolwarm')
plt.title('Financial Incentives by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Financial Incentives (USD)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.tight_layout()
plt.show()

# Emission Reduction and Jobs Created
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Type_of_Renewable_Energy', y='GHG_Emission_Reduction_tCO2e', data=grouped_df, palette='coolwarm')
plt.title('GHG Emission Reduction by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('GHG Emission Reduction (tCO2e)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.subplot(1, 2, 2)
sns.barplot(x='Type_of_Renewable_Energy', y='Jobs_Created', data=grouped_df, palette='coolwarm')
plt.title('Jobs Created by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Number of Jobs Created')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.tight_layout()
plt.show()
