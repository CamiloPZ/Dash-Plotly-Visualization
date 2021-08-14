import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
import dash_bootstrap_components as dbc

df = px.data.tips()
X = df.total_bill.values[:, None]
X_train, X_test, y_train, y_test = train_test_split(X, df.tip, random_state=42)

models = {
    "Regression": linear_model.LinearRegression,
    "Decision Tree": tree.DecisionTreeRegressor,
    "k-NN": neighbors.KNeighborsRegressor,
}

dff = px.data.gapminder()

fig = px.bar(
    dff,
    x="continent",
    y="pop",
    color="continent",
    animation_frame="year",
    animation_group="country",
    range_y=[0, 4000000000],
)

animations = {
    "Scatter": px.scatter(
        dff,
        x="gdpPercap",
        y="lifeExp",
        animation_frame="year",
        animation_group="country",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=55,
        range_x=[100, 100000],
        range_y=[25, 90],
    ),
    "Bar": px.bar(
        dff,
        x="continent",
        y="pop",
        color="continent",
        animation_frame="year",
        animation_group="country",
        range_y=[0, 4000000000],
    ),
}

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id="names",
                            value="day",
                            options=[
                                {"value": x, "label": x}
                                for x in ["smoker", "day", "time", "sex"]
                            ],
                            clearable=False,
                        ),
                        html.P("Values:"),
                        dcc.Dropdown(
                            id="values",
                            value="total_bill",
                            options=[
                                {"value": x, "label": x}
                                for x in ["total_bill", "tip", "size"]
                            ],
                            clearable=False,
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="pie-chart"),
                    ]
                ),
                dbc.Col(
                    [
                        html.P("Select an animation:"),
                        dcc.RadioItems(
                            id="selection",
                            options=[{"label": x, "value": x} for x in animations],
                            value="Scatter",
                        ),
                        dcc.Graph(id="graph"),
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="bar_ani", figure=fig),
                    ]
                ),
                dbc.Col([]),
                dbc.Col([]),
            ]
        ),
        html.P("Select Model:"),
        dcc.Dropdown(
            id="model-name",
            options=[{"label": x, "value": x} for x in models],
            value="Regression",
            clearable=False,
        ),
        dcc.Graph(id="graph_ani"),
    ]
)


@app.callback(Output("graph_ani", "figure"), [Input("selection", "value")])
def display_animated_graph(s):
    return animations[s]


@app.callback(
    Output("pie-chart", "figure"), [Input("names", "value"), Input("values", "value")]
)
def generate_chart(names, values):
    fig = px.pie(df, values=values, names=names)
    return fig


@app.callback(Output("graph", "figure"), [Input("model-name", "value")])
def train_and_display(name):
    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure(
        [
            go.Scatter(x=X_train.squeeze(), y=y_train, name="train", mode="markers"),
            go.Scatter(x=X_test.squeeze(), y=y_test, name="test", mode="markers"),
            go.Scatter(x=x_range, y=y_range, name="prediction"),
        ]
    )

    return fig


app.run_server(debug=True)
