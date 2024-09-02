import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dash import Dash, Output, Input, dcc, html
import plotly.express as px

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split





df = pd.read_csv("data.csv")


df['temperature range'] = 'Unknown'
df['humidity range'] = 'Unknown'
df['rainfall range'] = 'Unknown'

for i in range(0, 41, 10):  
    df['temperature range'] = np.where(
        (df['temperature'] >= i) & (df['temperature'] < i + 10),
        f'{i} - {i+10} °C',
        df['temperature range']
    )

# Define and assign humidity ranges (0 to 100, step of 20)
for i in range(0, 101, 20):  # Range from 14 to 100 with step of 20
    df['humidity range'] = np.where(
        (df['humidity'] >= i) & (df['humidity'] < i + 20),
        f'{i} - {i+20} %',
        df['humidity range']
    )

# Define and assign rainfall ranges (0 to 300, step of 50)
for i in range(0, 301, 50):  # Range from 20 to 300 with step of 50
    df['rainfall range'] = np.where(
        (df['rainfall'] >= i) & (df['rainfall'] < i + 50),
        f'{i} - {i+50} mm',
        df['rainfall range']
    )


##.....................prediction......................##

X = df.drop(['label', 'temperature range', 'humidity range', 'rainfall range'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

##......................dashboard.......................##


app = Dash(__name__)

temp_range = df['temperature range'].unique()
temp_range.sort()
humid_range = df['humidity range'].unique()
humid_range.sort()
'''rain_range = df['rainfall range'].unique()
rain_range.sort()''' #this  returns '0 - 50' in the last, because it sorts sorts strings lexicographically, not numerically.

rain_range = ['0 - 50 mm', '50 - 100 mm', '100 - 150 mm', '150 - 200 mm', '250 - 300 mm']
# there is no crops need '200- 250 mm', so I remove it.


app.layout = html.Div([
    
    html.H1(children = "Optimizing Agricultural Dashboard", style={'textAlign': 'center', 'color': 'black', 'font-size': '60px'}),
    

    html.Div([
        html.Div([
            
            html.H2(children="Select the Temperature Range", style={'color': 'black', 'textAlign': 'left'}),
            dcc.Dropdown(temp_range, value=temp_range[0], id='dropdown-temp-range', style={'width': '100%'})
            
        ], style={'width': '30%', 'padding': '10px'}),
        
        # Container for Humidity Range
        html.Div([
            
            html.H2(children="Select the Humidity Range", style={'color': 'black', 'textAlign': 'left'}),
            dcc.Dropdown(humid_range, value=humid_range[0], id='dropdown-humid-range', style={'width': '100%'})
            
        ], style={'width': '30%', 'padding': '10px'}),
        
        # Container for Rainfall Range
        html.Div([
            
            html.H2(children="Select the Rainfall Range", style={'color': 'black', 'textAlign': 'left'}),
            dcc.Dropdown(rain_range, value=rain_range[0], id='dropdown-rain-range', style={'width': '100%'})
            
        ], style={'width': '30%', 'padding': '10px'})
        
    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
    
    html.Div([
        
        dcc.Graph(id='Gragh-temp', style={'width' : '30%'}),
        dcc.Graph(id='Gragh-humid', style={'width' : '30%'}),
        dcc.Graph(id='Gragh-rain', style={'width' : '30%'})
        
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    
    html.H2(children="Select the Crop to show the Climatic Conditions", style={'textAlign': 'left', 'color': 'black'}),

    dcc.Dropdown(df.label.unique(), value='rice', id='dropdown-labels'),
    
    html.Div([
        
        dcc.Graph(id='Gragh-1', style={'width' : '30%'}),
        dcc.Graph(id='Gragh-2', style={'width' : '30%'}),
        dcc.Graph(id='Gragh-3', style={'width' : '30%'})
        
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    
    
    html.H2(children="Select from the Soil Conditions", style={'textAlign': 'left', 'color': 'black'}),
    dcc.Dropdown(['Nitrogen', 'Phosphorus', 'Potassium', 'PH'], value='Nitrogen', id='dropdown-soil-conditions'),
    dcc.Graph(id='Graph-soil-conditions'),
        
    #'''hint'''    
    #you can make the Prediction Section as the first section, to make each 'Input' and 'H3' in Div, and collect the all in one Div. (as you like!) 
    
    # The Prediction Section  
    html.H2(children="Enter the desired Agricultural Conditions", style={'textAlign': 'center', 'color': 'black', 'font-size': '40px'}),
    html.Div([
        html.H3(children="Nitrogen", style={ 'font-size': '20px', 'color': 'black', 'text-align': 'center', 'margin-right': '210px', 'margin-left': '10px'}),
        html.H3(children="Phosphorus", style={ 'font-size': '20px', 'color': 'black', 'text-align': 'center', 'margin-right': '180px'}),
        html.H3(children="Potassium", style={ 'font-size': '20px', 'color': 'black', 'text-align': 'center', 'margin-right': '190px'}),
        html.H3(children="PH", style={ 'font-size': '20px', 'color': 'black', 'text-align': 'center', 'margin-right': '255px'}),
        html.H3(children="Temperature", style={ 'font-size': '20px', 'color': 'black', 'text-align': 'center', 'margin-right': '175px'}),
        html.H3(children="Humidity", style={ 'font-size': '20px', 'color': 'black', 'text-align': 'center', 'margin-right': '200px'}),
        html.H3(children="Rainfall", style={ 'font-size': '20px', 'color': 'black', 'text-align': 'center', 'margin-right': '20px'})
        
    ], style={'display': 'flex'}),
     
    
    html.Div([
        
        #  N P K temperature humidity ph rainfall   
        
        dcc.Input(id='feature-1', type='text', placeholder='Nitrogen value', style={'hight': '50px', 'width': '190px', 'font-size': '20px', 'text-align': 'center'}),
        dcc.Input(id='feature-2', type='text', placeholder='Phosphorus value', style={'hight': '50px', 'width': '190px', 'font-size': '20px', 'text-align': 'center'}),
        dcc.Input(id='feature-3', type='text', placeholder='Potassium value', style={ 'hight': '50px', 'width': '190px', 'font-size': '20px', 'text-align': 'center'}),
        dcc.Input(id='feature-6', type='text', placeholder='PH value', style={'hight': '50px', 'width': '190px', 'font-size': '20px', 'text-align': 'center'}),
        dcc.Input(id='feature-4', type='text', placeholder='Temperature value', style={'hight': '50px', 'width': '190px', 'font-size': '20px', 'text-align': 'center'}),
        dcc.Input(id='feature-5', type='text', placeholder='Humidity value', style={'hight': '50px', 'width': '190px', 'font-size': '20px', 'text-align': 'center'}),
        dcc.Input(id='feature-7', type='text', placeholder='Rainfall value', style={'hight': '50px', 'width': '190px', 'font-size': '20px', 'text-align': 'center'}),
        
        
        
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    
    dcc.Graph(id='Graph-prediction')
    
    ])

# the colors for each label
colors=['#008080','#70a494','#b4c8a8','#f6edbd','#edbb8a','#de8a5a','#ca562c',  # fall
        '#009392','#39b185','#9ccb86','#e9e29c','#eeb479','#e88471','#cf597e',  # temps
        '#A16928','#bd925a','#d6bd8d','#edeac2','#b5c8b8','#79a7ac','#2887a1',  # earth
        '#798234'
        ] 

unique_labels = df['label'].unique()
color_map = {label: colors[i] for i, label in enumerate(unique_labels)}


@app.callback(
    
    Output('Gragh-temp', 'figure'),
    Input('dropdown-temp-range', 'value')
)
def update_temp_graph(temp_range):
    

    
    
    dff = df[df['temperature range'] == temp_range]
    dffgb = dff.groupby('label')['temperature'].agg(['count'])
    dffgb.reset_index(inplace=True)
    
    fig = px.pie(dffgb, values='count', names='label', hole=0.4, color='label', color_discrete_map=color_map)
    
    return fig

@app.callback(
    
    Output('Gragh-humid', 'figure'),
    Input('dropdown-humid-range', 'value')
)
def update_humid_graph(humid_range):
    
    dff = df[df['humidity range'] == humid_range]
    dffgb = dff.groupby('label')['humidity'].agg(['count'])
    dffgb.reset_index(inplace=True)
    
    fig = px.pie(dffgb, values='count', names='label', hole=0.4, color='label', color_discrete_map=color_map)
    
    return fig

@app.callback(
    
    Output('Gragh-rain', 'figure'),
    Input('dropdown-rain-range', 'value')
)
def update_rain_graph(rain_range):
    
    dff = df[df['rainfall range'] == rain_range]
    dffgb = dff.groupby('label')['rainfall'].agg(['count'])
    dffgb.reset_index(inplace=True)
    
    fig = px.pie(dffgb, values='count', names='label', hole=0.4, color='label', color_discrete_map=color_map)
    
    return fig

        

@app.callback(
    Output('Gragh-1', 'figure'),
    Output('Gragh-2', 'figure'),
    Output('Gragh-3', 'figure'),

    Input('dropdown-labels', 'value')
)

def update_graph(label):
    
   
    dff = df[df['label'] == label]
    
    '''
    for feature in ['temperature', 'humidity', 'rainfall']:

        fig = px.histogram(dff, x=feature)
        return fig
        
    The loop is not correctly returning separate figures for each graph. 
    Currently, it returns only the first figure three times.

    '''
    
    '''hint'''
    # we can round the values of the 'temperature', 'humidity', and 'rainfall' IF the decimal numbers is NOt impotant

    dff['temperature'] = round(dff['temperature'], 0)
    dff['humidity'] = round(dff['humidity'], 0)
    dff['rainfall'] = round(dff['rainfall'], 0)
   
    
    fig1 = px.histogram(dff, x='temperature',  color_discrete_sequence=['#008080'], labels={'temperature': 'Temperature °C '}, title=f"Temperature for {label} ({len(dff)})") # 'nbins' attribute to set the number of bins.
    fig2 = px.histogram(dff, x='humidity',  color_discrete_sequence=['#70a494'],labels={'humidity': 'Humidity %'}, title=f"Humidity for {label} ({len(dff)})")
    fig3 = px.histogram(dff, x='rainfall',  color_discrete_sequence=['#b4c8a8'], labels={'rainfall': 'Rainfall mm'}, title=f"Rainfall for {label} ({len(dff)})")
    
    fig1.update_layout(bargap=0.2)
    fig2.update_layout(bargap=0.2)
    fig3.update_layout(bargap=0.2)
    
    
    return fig1, fig2, fig3



@app.callback(
    
    Output('Graph-soil-conditions', 'figure'),
    Input('dropdown-soil-conditions', 'value')
)
def update_soil_conditions_graph(soil_conditions):
    
    soil = None
    if soil_conditions == 'Nitrogen':
        soil = 'N'
        soil_conditions+= ' (ppm)'
    elif soil_conditions == 'Phosphorus':
        soil = 'P'
        soil_conditions+= ' (ppm)'
    elif soil_conditions == 'Potassium':
        soil = 'K'
        soil_conditions+= ' (ppm)'
    elif soil_conditions == 'PH':
        soil = 'ph'
    
    dff = df.groupby('label')[soil].agg(['mean']).reset_index()
    dff.columns = ['label', soil]
    
    

    # to sort the values of dff['label'] to be in the same order of df['label']
    unique_labels = df['label'].unique()
    dff['label'] = pd.Categorical(dff['label'], categories=unique_labels, ordered=True)
    dff = dff.sort_values('label').reset_index(drop=True)
   
    
    fig = px.bar(dff, x='label', y=soil, labels={soil: soil_conditions, 'label': 'The Crops'}, color='label', color_discrete_map=color_map)
    
    '''hint'''      
    # if I removed ' color='label' ' the figure will be the the same color (the first color in the 'color_map')
          
    fig.update_layout(bargap=0.2)
    
    return fig

@app.callback(
    
    Output('Graph-prediction', 'figure'),
    Input('feature-1', 'value'),
    Input('feature-2', 'value'),
    Input('feature-3', 'value'),
    Input('feature-4', 'value'),
    Input('feature-5', 'value'),
    Input('feature-6', 'value'),
    Input('feature-7', 'value')
)

def update_pediction_graph(Nitrogen, Phosphorus, Potassium, PH, Temperature, Humidity,  Rainfall):
    
    inputs = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, PH, Rainfall]
    inputs = [float(i) if i is not None and i != '' else 0.0 for i in inputs] # Convert None to 0.0 or handle it as needed
    
    prediction = lr.predict_proba([inputs])
    labels = lr.classes_
    dff = pd.DataFrame({'label': labels, 'prediction': prediction[0]})
    dff.sort_values('prediction', ascending=False, inplace=True)
    
    fig = px.bar(dff, x='prediction', y='label', color='label', color_discrete_map=color_map)
    fig.update_layout(bargap=0.4, height=800)
    fig.update_xaxes(range=[0, 1])
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
