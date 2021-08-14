from re import TEMPLATE, template
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
from dash_extensions import Lottie
import plotly.graph_objects as go
import numpy as np


np.random.seed(1)

TEMPLATE = "plotly_dark"

df_i = px.data.iris()
fig_i = px.scatter_3d(
    df_i, x="sepal_length", y="sepal_width", z="petal_width", color="species"
)
fig_i.update_layout(template=TEMPLATE)

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

# Create traces
fig_l = go.Figure()
fig_l.add_trace(go.Scatter(x=random_x, y=random_y0, mode="lines", name="lines"))
fig_l.add_trace(
    go.Scatter(x=random_x, y=random_y1, mode="lines+markers", name="lines+markers")
)
fig_l.add_trace(go.Scatter(x=random_x, y=random_y2, mode="markers", name="markers"))

fig_l.update_layout(template=TEMPLATE)

# Add data
month = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
high_2000 = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3]
low_2000 = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9]
high_2007 = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0]
low_2007 = [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6]
high_2014 = [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9]
low_2014 = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]

fig_l2 = go.Figure()
# Create and style traces
fig_l2.add_trace(
    go.Scatter(
        x=month, y=high_2014, name="High 2014", line=dict(color="firebrick", width=4)
    )
)
fig_l2.add_trace(
    go.Scatter(
        x=month, y=low_2014, name="Low 2014", line=dict(color="royalblue", width=4)
    )
)
fig_l2.add_trace(
    go.Scatter(
        x=month,
        y=high_2007,
        name="High 2007",
        line=dict(
            color="firebrick", width=4, dash="dash"
        ),  # dash options include 'dash', 'dot', and 'dashdot'
    )
)
fig_l2.add_trace(
    go.Scatter(
        x=month,
        y=low_2007,
        name="Low 2007",
        line=dict(color="royalblue", width=4, dash="dash"),
    )
)
fig_l2.add_trace(
    go.Scatter(
        x=month,
        y=high_2000,
        name="High 2000",
        line=dict(color="firebrick", width=4, dash="dot"),
    )
)
fig_l2.add_trace(
    go.Scatter(
        x=month,
        y=low_2000,
        name="Low 2000",
        line=dict(color="royalblue", width=4, dash="dot"),
    )
)

# Edit the layout
fig_l2.update_layout(
    title="Average High and Low Temperatures in New York",
    xaxis_title="Month",
    yaxis_title="Temperature (degrees F)",
)

fig_l2.update_layout(template=TEMPLATE)

url_ytb = "https://assets2.lottiefiles.com/packages/lf20_JoLzcd.json"
link_ytb = "https://www.youtube.com/channel/UCAlUNyWsCyJeEMRnKKeWsxg"

url_linke = "https://assets8.lottiefiles.com/packages/lf20_yNYxCH.json"
link_linke = "https://www.linkedin.com/in/camilo-d-vinchi-poma-zamudio-142711139/"

url_mail = "https://assets4.lottiefiles.com/packages/lf20_odef4xnr.json"
link_mail = "mailto:cpomaz@uni.pe?subject=Mail from our Website"

url_git = "https://assets8.lottiefiles.com/packages/lf20_Cko7Sr.json"
link_git = "https://github.com/CamiloPZ"

url_face = "https://assets7.lottiefiles.com/private_files/lf30_pb3we3yk.json"
link_face = "https://www.facebook.com/CamiloPomaZ/"

url_inst = "https://assets7.lottiefiles.com/private_files/lf30_igdzkfxv.json"
link_insta = "https://www.instagram.com/camilo_poma/"

options = dict(
    loop=True,
    autoplay=True,
    rendererSettings=dict(preserveAspectRatio="xMidYMid slice"),
)

colorscale = [[0, "gold"], [0.5, "mediumturquoise"], [1, "lightsalmon"]]
fig_col = go.Figure(
    data=go.Contour(
        z=[
            [10, 10.625, 12.5, 15.625, 20],
            [5.625, 6.25, 8.125, 11.25, 15.625],
            [2.5, 3.125, 5.0, 8.125, 12.5],
            [0.625, 1.25, 3.125, 6.25, 10.625],
            [0, 0.625, 2.5, 5.625, 10],
        ],
        colorscale=colorscale,
    )
)

fig_col.update_layout(template=TEMPLATE)

df = px.data.tips()
X = df.total_bill.values[:, None]
X_train, X_test, y_train, y_test = train_test_split(X, df.tip, random_state=42)

fig_hist = px.histogram(
    df,
    x="total_bill",
    y="tip",
    color="sex",
    marginal="box",  # or violin, rug
    hover_data=df.columns,
)
fig_hist.update_layout(template=TEMPLATE)
models = {
    "Regression": linear_model.LinearRegression,
    "Decision Tree": tree.DecisionTreeRegressor,
    "k-NN": neighbors.KNeighborsRegressor,
}

fig_caja = px.box(
    df,
    x="time",
    y="total_bill",
    color="smoker",
    notched=True,  # used notched shape
    title="Box plot of total bill",
    hover_data=["day"],  # add day column to hover data
)
fig_caja.update_layout(template=TEMPLATE)

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

fig.update_layout(template=TEMPLATE)

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
    ).update_layout(template=TEMPLATE),
    "Bar": px.bar(
        dff,
        x="continent",
        y="pop",
        color="continent",
        animation_frame="year",
        animation_group="country",
        range_y=[0, 4000000000],
    ).update_layout(template=TEMPLATE),
}

# animations.update_layout(template=TEMPLATE)

app = dash.Dash(
    __name__,
    title="Ejemplos Dash",
    external_stylesheets=[dbc.themes.DARKLY],
)

app.layout = html.Div(
    [
        html.Br(),
        html.H1(
            "Ejemplos de gráficos interactivos",
            style={
                "textAlign": "center",
                "font-family": "Times New Roman",
            },
        ),
        html.Br(),
        dbc.CardDeck(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    Lottie(
                                        options=options,
                                        width="52%",
                                        height="67%",
                                        url=url_ytb,
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.CardLink("Youtube", href=link_ytb),
                                        html.H2(
                                            id="content-connections", children="300"
                                        ),
                                    ],
                                    style={
                                        "textAlign": "center",
                                        "font-family": "Comic Sans MS",
                                    },
                                ),
                            ]
                        ),
                    ],
                    xs=12,
                    md=4,
                    sm=6,
                    lg=2,
                    xl=2,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    Lottie(
                                        options=options,
                                        width="52%",
                                        height="32%",
                                        url=url_linke,
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.CardLink("LinkedIn", href=link_linke),
                                        html.H2(id="content-companies", children="200"),
                                    ],
                                    style={
                                        "textAlign": "center",
                                        "font-family": "Comic Sans MS",
                                    },
                                ),
                            ]
                        ),
                    ],
                    xs=12,
                    md=4,
                    sm=6,
                    lg=2,
                    xl=2,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    Lottie(
                                        options=options,
                                        width="53%",
                                        height="25%",
                                        url=url_mail,
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.CardLink("Mail", href=link_mail),
                                        html.H2(id="content-msg-in", children="4"),
                                    ],
                                    style={
                                        "textAlign": "center",
                                        "font-family": "Comic Sans MS",
                                    },
                                ),
                            ]
                        ),
                    ],
                    xs=12,
                    md=4,
                    sm=6,
                    lg=2,
                    xl=2,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    Lottie(
                                        options=options,
                                        width="53%",
                                        height="53%",
                                        url=url_git,
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.CardLink("GitHub", href=link_git),
                                        html.H2(id="content-msg-out", children="1"),
                                    ],
                                    style={
                                        "textAlign": "center",
                                        "font-family": "Comic Sans MS",
                                    },
                                ),
                            ]
                        ),
                    ],
                    xs=6,
                    md=4,
                    sm=6,
                    lg=2,
                    xl=2,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    Lottie(
                                        options=options,
                                        width="54%",
                                        height="25%",
                                        url=url_face,
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.CardLink("Facebook", href=link_face),
                                        html.H2(id="content-reactions", children="30"),
                                    ],
                                    style={
                                        "textAlign": "center",
                                        "font-family": "Comic Sans MS",
                                    },
                                ),
                            ]
                        ),
                    ],
                    xs=12,
                    md=4,
                    sm=6,
                    lg=2,
                    xl=2,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    Lottie(
                                        options=options,
                                        width="55%",
                                        height="85%",
                                        url=url_inst,
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.CardLink("Instagram", href=link_insta),
                                        html.H2(
                                            id="content-reactions2", children="100"
                                        ),
                                    ],
                                    style={
                                        "textAlign": "center",
                                        "font-family": "Comic Sans MS",
                                    },
                                ),
                            ]
                        ),
                    ],
                    xs=12,
                    md=4,
                    sm=6,
                    lg=2,
                    xl=2,
                ),
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="caja1", figure=fig_caja),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="caja2", figure=fig_col),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="caja3", figure=fig_i),
                    ]
                ),
            ],
            justify="around",
            align="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="line1", figure=fig_l),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="line2", figure=fig_l2),
                    ]
                ),
            ],
            justify="around",
            align="center",
        ),
        html.Br(),
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
                        dcc.Graph(id="pie-chart"),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="bar_ani", figure=fig_hist),
                    ]
                ),
            ],
            className="dbc_dark",
            justify="around",
            align="center",
        ),
        html.Br(),
        dbc.Row(
            [
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
                # dbc.Col([]),
                # dbc.Col([]),
            ],
            justify="around",
            align="center",
        ),
        html.Br(),
        html.P("Select Model:"),
        dcc.Dropdown(
            id="model-name",
            options=[{"label": x, "value": x} for x in models],
            value="Regression",
            clearable=False,
        ),
        dcc.Graph(id="graph_ani"),
    ],
    className="dbc_dark",
)


@app.callback(Output("graph", "figure"), [Input("selection", "value")])
def display_animated_graph(s):
    return animations[s]


@app.callback(
    Output("pie-chart", "figure"), [Input("names", "value"), Input("values", "value")]
)
def generate_chart(names, values):
    fig = px.pie(df, values=values, names=names)
    fig.update_layout(template=TEMPLATE)
    return fig


@app.callback(Output("graph_ani", "figure"), [Input("model-name", "value")])
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
    fig.update_layout(template=TEMPLATE)

    return fig


app.run_server(debug=True)
