import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import sklearn
import os
import re


df = pd.read_csv("/Users/dhartipatel/Documents/GitHub/PRODIGY_DS_02/energy_dataset_.csv")
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
#The installed capacity for different types of renewable energy sources (Solar, Wind, Hydroelectric, Geothermal, Biomass, Tidal, Wave) appears to be relatively similar, with each type having an installed capacity around 500 MW. There is no single dominant energy source in terms of installed capacity.


# Energy Production vs. Consumption
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Energy_Production_MWh', y='Energy_Consumption_MWh', hue='Type_of_Renewable_Energy', palette='viridis', data=df)
plt.title('Energy Production vs. Consumption')
plt.xlabel('Energy Production (MWh)')
plt.ylabel('Energy Consumption (MWh)')
plt.legend(title='Type of Renewable Energy', labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()
#The scatter plot shows a very dense and widespread distribution of points. This indicates a significant variation in energy production and consumption values among the different renewable energy sources. There is no clear trend or linear relationship between energy production and consumption for the different types of renewable energy, suggesting that production and consumption are not directly correlated in this dataset.


## Storage Efficiency of Different Types of Renewable Energy
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type_of_Renewable_Energy', y='Storage_Efficiency_Percentage', data=df, palette='viridis')
plt.title('Storage Efficiency of Different Types of Renewable Energy')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Storage Efficiency (%)')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()
#The box plot reveals that storage efficiency percentages for different types of renewable energy are relatively consistent, mostly falling between 60% and 90%. Solar energy has a slightly lower median storage efficiency compared to the others, while Tidal and Wave energy show a broader range of efficiency.


## Initial Investment vs. GHG Emission Reduction
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Initial_Investment_USD', y='GHG_Emission_Reduction_tCO2e', hue='Type_of_Renewable_Energy', palette='viridis', data=df)
plt.title('Initial Investment vs. GHG Emission Reduction')
plt.xlabel('Initial Investment (USD)')
plt.ylabel('GHG Emission Reduction (tCO2e)')
plt.legend(title='Type of Renewable Energy', labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()
#The scatter plot shows a broad distribution of data points, indicating that there is a wide range of initial investment and corresponding greenhouse gas (GHG) emission reductions across the different types of renewable energy. There is no clear pattern or trend that suggests a direct relationship between the amount of initial investment and the reduction in GHG emissions.


## Number of Jobs Created by Different Types of Renewable Energy
plt.figure(figsize=(10, 6))
sns.barplot(x='Type_of_Renewable_Energy', y='Jobs_Created', data=df, ci=None, palette='viridis')
plt.title('Number of Jobs Created by Different Types of Renewable Energy')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Number of Jobs Created')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()
#The bar plot indicates that the number of jobs created by different types of renewable energy is relatively similar, with each type creating around 2500 jobs. This suggests that renewable energy projects, regardless of the type, tend to have a similar impact on job creation


# Further Analysis
## 1. Correlation Analysis
# Calculate the correlation matrix

plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# correlation_matrix = df.corr()

# # Plot the correlation matrix
# plt.figure(figsize=(15, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Matrix')
# plt.show()
#The correlation matrix shows that most variables have very weak correlations with each other. This indicates that the dataset's features are largely independent.
##Notable Correlations:

    #1. There is a slight positive correlation between Installed_Capacity_MW and GHG_Emission_Reduction_tCO2e.
    #2. Initial_Investment_USD has a very weak positive correlation with Energy_Production_MWh.
    #3. Financial_Incentives_USD has a very weak positive correlation with GHG_Emission_Reduction_tCO2e.



## 2. Detailed Breakdown by Energy Type
# Group by type of renewable energy and calculate mean values
grouped_df = df.groupby('Type_of_Renewable_Energy').mean().reset_index()

# Plot installed capacity, energy production, and energy consumption
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.barplot(x='Type_of_Renewable_Energy', y='Installed_Capacity_MW', data=grouped_df, palette='viridis')
plt.title('Installed Capacity by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Installed Capacity (MW)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.subplot(1, 3, 2)
sns.barplot(x='Type_of_Renewable_Energy', y='Energy_Production_MWh', data=grouped_df, palette='viridis')
plt.title('Energy Production by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Energy Production (MWh)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.subplot(1, 3, 3)
sns.barplot(x='Type_of_Renewable_Energy', y='Energy_Consumption_MWh', data=grouped_df, palette='viridis')
plt.title('Energy Consumption by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Energy Consumption (MWh)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.tight_layout()
plt.show()

#The installed capacity, energy production, and energy consumption are relatively consistent across different types of renewable energy. This uniformity suggests a balanced distribution in the dataset.

    #1. Installed Capacity: Hydroelectric seems to have the highest average installed capacity.
    #2. Energy Production: All types of renewable energy have similar levels of energy production.
    #3. Energy Consumption: Similarly, the energy consumption levels are also quite uniform across different renewable energy types.



## 3. Efficiency Improvement Opportunities

# Group by type of renewable energy and calculate summary statistics for storage efficiency
efficiency_summary = df.groupby('Type_of_Renewable_Energy')['Storage_Efficiency_Percentage'].describe()
print(efficiency_summary)

# Plot storage efficiency distributions
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type_of_Renewable_Energy', y='Storage_Efficiency_Percentage', data=df, palette='viridis')
plt.title('Storage Efficiency by Type of Renewable Energy')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Storage Efficiency (%)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])
plt.show()

#The boxplot shows that storage efficiency percentages for different types of renewable energy sources have a broad range, generally between 60% and 90%.
    #1. Median Efficiency: All types of renewable energy have a median storage efficiency around 75% to 85%.
    #2. Variability: Solar and Tidal energy types show slightly more variability in storage efficiency.



## 4. Additional Analysis: Financial Metrics

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Type_of_Renewable_Energy', y='Initial_Investment_USD', data=grouped_df, palette='viridis')
plt.title('Initial Investment by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Initial Investment (USD)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.subplot(1, 2, 2)
sns.barplot(x='Type_of_Renewable_Energy', y='Financial_Incentives_USD', data=grouped_df, palette='viridis')
plt.title('Financial Incentives by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Financial Incentives (USD)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.tight_layout()
plt.show()

#Initial investment and financial incentives are quite similar across different types of renewable energy.
    #1. Initial Investment: The investment required for each type of renewable energy does not vary significantly, with all types requiring roughly similar amounts of initial investment.
    #2. Financial Incentives: The financial incentives are also similar across the board, indicating a uniform approach to incentivizing various types of renewable energy.



## 5. Emission Reduction and Jobs Created
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Type_of_Renewable_Energy', y='GHG_Emission_Reduction_tCO2e', data=grouped_df, palette='viridis')
plt.title('GHG Emission Reduction by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('GHG Emission Reduction (tCO2e)')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.subplot(1, 2, 2)
sns.barplot(x='Type_of_Renewable_Energy', y='Jobs_Created', data=grouped_df, palette='viridis')
plt.title('Jobs Created by Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Number of Jobs Created')
plt.xticks(ticks=range(7), labels=['Solar', 'Wind', 'Hydroelectric', 'Geothermal', 'Biomass', 'Tidal', 'Wave'])

plt.tight_layout()
plt.show()

#Both GHG emission reduction and the number of jobs created are relatively consistent across different types of renewable energy.
    #1. GHG Emission Reduction: The reduction in greenhouse gas emissions is quite uniform, with each type contributing similarly.
    #2. Jobs Created: The number of jobs created by each type of renewable energy is also quite similar, suggesting that renewable energy projects generally create comparable employment opportunities regardless of the type.


## Conclusion
#The dataset appears to be quite balanced in terms of the installed capacity, energy production, energy consumption, storage efficiency, financial aspects, emission reductions, and job creation across different types of renewable energy. There are no strong correlations between most of the variables, indicating that each feature has a unique contribution to the dataset.