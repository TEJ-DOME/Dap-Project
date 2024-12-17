#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import seaborn as sns


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import plotly.express as px


# In[6]:


dataset = pd.read_csv(r"D:\tej\Python\countries-table.csv")


# In[7]:


dataset.info()


# In[8]:


dataset.head(234)


# In[9]:


dataset.columns


# In[10]:


def format_population(value):
    if value >= 1_000_000_000:  # Billion
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:  # Million
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:  # Thousand
        return f"{value / 1_000:.2f}K"
    else:
        return str(value)

formatted_pop1980 = [format_population(pop) for pop in dataset['pop1980']]
formatted_pop2000 = [format_population(pop) for pop in dataset['pop2000']]
formatted_pop2010 = [format_population(pop) for pop in dataset['pop2010']]
formatted_pop2022 = [format_population(pop) for pop in dataset['pop2022']]
formatted_pop2023 = [format_population(pop) for pop in dataset['pop2023']]
formatted_pop2030 = [format_population(pop) for pop in dataset['pop2030']]
formatted_pop2050 = [format_population(pop) for pop in dataset['pop2050']]


# In[15]:


top_10_countries = dataset.sort_values(by="pop2023", ascending=False).head(10)
top_10_countries['formatted_pop'] = top_10_countries['pop2023'].apply(format_population)
fig = px.bar(top_10_countries, x="country", y="pop2023", 
             title="Top 10 Countries by Population 2023", 
             labels={"pop2023": "Population", "country": "Country"},height = 500)

fig.update_traces(text=top_10_countries['formatted_pop'], 
                  textposition='outside',
                  texttemplate='%{text}',
                  insidetextanchor='start')

fig.show()


# In[12]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

top_5_countries = dataset.sort_values(by='pop2023', ascending=False).head(5)

df_long = top_5_countries.melt(id_vars=['country'], 
                                value_vars=['pop1980', 'pop2000', 'pop2010', 'pop2022', 'pop2023'],
                                var_name='year', value_name='population')

df_long['formatted_pop'] = df_long['population'].apply(format_population)

fig = make_subplots(
    rows=1, 
    cols=5, 
    subplot_titles=top_5_countries['country'],
    shared_yaxes=True,
)

for i, country in enumerate(top_5_countries['country']):
    country_data = df_long[df_long['country'] == country] 
    
    fig.add_trace(
        go.Bar(
            x=country_data['year'],
            y=country_data['population'],
            name=country,
            text=country_data['formatted_pop'],
            textposition='outside',
            texttemplate='%{text}',
            insidetextanchor='start',
            textfont=dict(size=8)
        ),
        row=1, col=i+1 
    )

fig.update_layout(
    title="Top 5 Countries by Population (1980-2023)",
    showlegend=False,
    height=500,
    width=1150,
    barmode='group', 
    xaxis_title="Year",
    yaxis_title="Population",
    margin=dict(t=80, b=60, l=40, r=40), 
    font=dict(size=12)
)

for i in range(1, 6):
    fig.update_xaxes(
        tickmode='array',
        tickvals=['pop1980', 'pop2000', 'pop2010', 'pop2022', 'pop2023'],
        ticktext=['1980', '2000', '2010', '2022', '2023'],
        row=1, col=i,
        tickangle=-45, 
        tickfont=dict(size=10)
    )

fig.update_yaxes(
    tickfont=dict(size=12) 
)

fig.show()


# In[13]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots


data = {
    'country': ['India', 'China', 'United States'],
    'pop1980': [696828385, 982372466, 223140018],
    'pop2000': [1059633675, 1264099069, 282398554],
    'pop2010': [1240613620, 1348191368, 311182845],
    'pop2022': [1417173173, 1425887337, 338289857],
    'pop2023': [1428627663, 1425671352, 339996563],
    'pop2030': [1514994080, 1415605906, 352162301], 
    'pop2050': [1670490596, 1312636325, 375391963], 
}


df = pd.DataFrame(data)


def calc_growth_rate(start_population, end_population):
    return ((end_population - start_population) / start_population) * 100

years = ['pop1980', 'pop2000', 'pop2010', 'pop2022', 'pop2023', 'pop2030', 'pop2050']

growth_data = {
    'country': [],
    'year': [],
    'growth_rate': []
}

for country in df['country']:
    for i in range(len(years) - 1):
        start_year = years[i]
        end_year = years[i + 1]
        
        start_population = df.loc[df['country'] == country, start_year].values[0]
        end_population = df.loc[df['country'] == country, end_year].values[0]
        
        growth_rate = calc_growth_rate(start_population, end_population)
        
        growth_data['country'].append(country)
        growth_data['year'].append(f"{start_year[3:]} to {end_year[3:]}") 
        growth_data['growth_rate'].append(growth_rate)

growth_df = pd.DataFrame(growth_data)

fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=df['country'].values, 
    shared_yaxes=True,
    vertical_spacing=0.1
)

for i, country in enumerate(df['country']):
    country_data = growth_df[growth_df['country'] == country]
    fig.add_trace(
        go.Scatter(
            x=country_data['year'],
            y=country_data['growth_rate'],
            mode='lines+markers', 
            name=country,
            line=dict(width=3),
            marker=dict(size=8),
            text=country_data['growth_rate'].round(3),
            textposition='top center'
        ),
        row=1, col=i+1 
    )

fig.update_layout(
    title="Population Growth Rates of 3 Countries",
    xaxis_title="Year Transition",
    yaxis_title="Growth Rate (%)",
    template="plotly_dark",
    height=400,
    width=1100,
    showlegend=False
)

for i in range(1, 4):
    fig.update_xaxes(
        tickmode='array',
        tickvals=growth_df['year'][::2],
        ticktext=growth_df['year'][::2],
        row=1, col=i,
        tickangle=+15, 
        tickfont=dict(size=10) 
    )

fig.show()


# In[16]:


data = {
    'country': ['India', 'China', 'United States'],
    'land_area_km2': [2973190, 9424702, 9147420],
    'population': [1428627663, 1425671352, 339996563],
}

df = pd.DataFrame(data)

fig = px.choropleth(
    df, 
    locations='country', 
    locationmode='country names',
    color='land_area_km2', 
    hover_name='country',
    hover_data={'land_area_km2': True, 'population': True},
    color_continuous_scale='Viridis',
    labels={'land_area_km2': 'Land Area (km²)', 'population': 'Population'},
    title="Land Area and Population of India, China, and United States",
)

fig.update_geos(showcoastlines=True, coastlinecolor="Black", projection_type="natural earth")
fig.update_layout(
    geo=dict(showland=True, landcolor="lightgray"),
    title="Land Area and Population of India, China, and United States",
)

fig.show()


# In[ ]:





# In[ ]:




