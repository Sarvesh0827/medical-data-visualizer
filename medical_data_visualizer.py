import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importing the data
df = pd.read_csv('medical_examination.csv')

# 2. Adding an overweight column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalizing cholesterol and glucose values (making 0 always good and 1 always bad)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Function to draw the categorical plot
def draw_cat_plot():
    # 5. Creating the DataFrame for the cat plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat the data by cardio and feature values
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Drawing the categorical plot using seaborn's catplot
    fig = sns.catplot(data=df_cat, x='variable', hue='value', col='cardio', kind='bar', height=5, aspect=1.2).fig

    # 8. Save the figure
    fig.savefig('catplot.png')
    return fig


# 10. Function to draw the heat map
def draw_heat_map():
    # 11. Cleaning the data by filtering out incorrect data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12. Calculating the correlation matrix
    corr = df_heat.corr()

    # 13. Generating a mask for the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Setting up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15. Plotting the heatmap using seaborn's heatmap function
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', cbar_kws={'shrink': .8}, ax=ax)

    # 16. Saving the figure
    fig.savefig('heatmap.png')
    return fig
