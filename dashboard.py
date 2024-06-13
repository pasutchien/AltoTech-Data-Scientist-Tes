from plotly.data import gapminder
from dash import dcc, html, Dash, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import mean_squared_error


df = pd.read_csv('done.csv')
css = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css", ]
app = Dash(name="Alto Energy Zero", external_stylesheets=css)
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month

df2 = pd.read_csv('savings.csv')
df2['datetime'] = pd.to_datetime(df2['datetime'])
df2['month'] = df2['datetime'].dt.month

def create_table():
    fig = go.Figure(data=[go.Table(
        header=dict(values=df.columns, align='left'),
        cells=dict(values=df.values.T, align='left'))
    ]
    )
    fig.update_layout(paper_bgcolor="#e5ecf6", margin={"t":0, "l":0, "r":0, "b":0}, height=700)
    return fig
def create_line_graph(model = "XGBoost(My chosen model)", month = 7):
    monthlydata = df[df['datetime'].dt.month == month].copy()
    monthlydata.loc[:,'day'] = monthlydata['datetime'].dt.day
    monthlydata.loc[:,'hour'] = monthlydata['datetime'].dt.hour
    if month == 10:
        temp = monthlydata[monthlydata['datetime'] <  '2022-10-11'].copy()
        mse = mean_squared_error(temp['power'],temp[model])
    if month == 11 or month == 12:
        mse = "No data"
    else:
        mse = mean_squared_error(monthlydata['power'],monthlydata[model])

    
    # Create a new column for day:hour
    monthlydata.loc[:,'day_hour'] = monthlydata['day'].astype(str) + ':' + monthlydata['hour'].astype(str).str.zfill(2)
    fig = px.line(monthlydata, x="day_hour", y=["power", model],
                  title=f'Actual Power VS Predicted Power, MSE of this month = {mse}', color_discrete_map={'power': 'blue',model: 'red'})
    
    fig.update_layout(xaxis_title='day:hour', yaxis_title='Power', legend_title='Data', # Increase the width to create a horizontal scroll bar
            xaxis=dict(
            rangeslider=dict(visible=True),  # Enable range slider for horizontal scroll
            type="category"
        )
    )
    return fig
def create_line_graph_difference(col = 'cumulative_energy VS cumulative_predicted_energy',month = 10):
    col1, col2 = col.split(' VS ')
    monthly = df2[df2['datetime'].dt.month == month].copy()
    monthly.loc[:,'day'] = monthly['datetime'].dt.day
    monthly.loc[:,'hour'] = monthly['datetime'].dt.hour
    monthly.loc[:,'day_hour'] = monthly['day'].astype(str) + ':' + monthly['hour'].astype(str).str.zfill(2)
    monthly.loc[:, 'diff'] = monthly[col2] - monthly[col1]
    monthly.loc[:,'saving(%)'] = (monthly['diff']/monthly[col2])*100
    fig = px.line(monthly, x="day_hour", y=[col1, col2,'diff','saving(%)'],
                  title=f'{col1} VS {col2}', color_discrete_map={col1: 'blue',col2: 'green','diff':'red','saving(%)':'black'})
    fig.update_layout(xaxis_title='day:hour', yaxis_title='Energy(kWh)', legend_title='Energy_Comparison')
    return fig
def create_savings_graph(col = 'cumulative_predicted_energy',month = 10):
    def electricity_critera(value):
        if value <= 150:
            return value * 3.25
        elif value > 150 and value <= 400:
            return 487.5 + (value-150) * 4.22
        elif value > 400:
            return 1542.5 + (value-400) * 4.42
    monthly = df2[df2['datetime'].dt.month == month].copy()
    first_index = monthly.index[0]
    if month == 10:
        start_real = 0
        start_baseline = 0
    else:
        start_real = df2.loc[first_index-1,'cumulative_energy']
        start_baseline = df2.loc[first_index-1,col]
    monthly.loc[:,'day'] = monthly['datetime'].dt.day
    monthly.loc[:,'hour'] = monthly['datetime'].dt.hour
    monthly.loc[:,'day_hour'] = monthly['day'].astype(str) + ':' + monthly['hour'].astype(str).str.zfill(2)
    monthly.loc[:,'cost_with_Alto'] = (monthly['cumulative_energy']-start_real).apply(electricity_critera)
    monthly.loc[:,'baseline_cost'] = (monthly[col]-start_baseline).apply(electricity_critera)
    monthly.loc[:, 'savings'] = monthly['baseline_cost'] - monthly['cost_with_Alto']
    fig = px.line(monthly, x="day_hour", y=['cost_with_Alto', 'baseline_cost','savings'],
                  title=f'Baseline Cost VS Cost after Alto Energy Zero', color_discrete_map={'cost_with_Alto': 'blue','baseline_cost': 'green','savings':'red'})
    fig.update_layout(xaxis_title='day:hour', yaxis_title='Cost(Baht)', legend_title='Savings')
    return fig

#power,predicted_energy,cumulative_energy,cumulative_predicted_energy,cumulative_predicted_energy_lower_bound,cumulative_predicted_energy_upper_bound
models = [col for col in df.columns if col not in ["datetime","power","month"]]
months = df["month"].unique()
models_dd = dcc.Dropdown(id="models_dd", options=models, value="XGBoost(My chosen model)",clearable=False)
months_dd = dcc.Dropdown(id="months_dd", options=months, value=7,clearable=False)

comparison = ['cumulative_energy VS cumulative_predicted_energy', 'cumulative_energy VS cumulative_predicted_energy_lower_bound', 'cumulative_energy VS cumulative_predicted_energy_upper_bound']
months2 = df2["month"].unique()
comparison_dd = dcc.Dropdown(id="comparison_dd", options=comparison, value='cumulative_energy VS cumulative_predicted_energy',clearable=False)
months2_dd = dcc.Dropdown(id="months2_dd", options=months2, value=10,clearable=False)

saving_comparison = ['cumulative_predicted_energy','cumulative_predicted_energy_lower_bound','cumulative_predicted_energy_upper_bound']
months3 = df2["month"].unique()
saving_comparison_dd = dcc.Dropdown(id="saving_comparison_dd", options=saving_comparison, value='cumulative_predicted_energy',clearable=False)
months3_dd = dcc.Dropdown(id="months3_dd", options=months3, value=10,clearable=False)

app.layout = html.Div([
    html.Div([
        html.H1("Alto Energy Zero", className="text-center fw-bold m-2"),
        html.Br(),
        dcc.Tabs([
            dcc.Tab([html.Br(),
                     dcc.Graph(id="dataset", figure=create_table())], label="Dataset"),
            dcc.Tab([html.Br(), "Model", models_dd, "Month", months_dd, html.Br(),
                     dcc.Graph(id="Model")], label="Model"),
            dcc.Tab([html.Br(), "Comparison", comparison_dd, "Month", months2_dd, html.Br(),
                     dcc.Graph(id="Energy")], label="Energy"),
            dcc.Tab([html.Br(), "Baseline Energy", saving_comparison_dd, "Month", months3_dd, html.Br(),
                     dcc.Graph(id="Savings")], label="Savings")
        ])
    ], className="col-8 mx-auto"),
], style={"background-color": "#e5ecf6", "height": "100vh"})

@callback(Output("Model", "figure"), [Input("models_dd", "value"), Input("months_dd", "value"),])
def update_prediction_model(model, month):
    return create_line_graph(model, month)
@callback(Output("Energy", "figure"), [Input("comparison_dd", "value"), Input("months2_dd", "value"),])
def update_energy_comparison(col, month):
    return create_line_graph_difference(col, month)
@callback(Output("Savings", "figure"), [Input("saving_comparison_dd", "value"), Input("months3_dd", "value"),])
def update_saving_comparison(col, month):
    return create_savings_graph(col, month)
if __name__ == "__main__":
    app.run(debug=False)
