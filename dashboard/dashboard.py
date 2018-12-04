import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

df = pd.DataFrame()

df['city'] = ["Chicago", "Atlanta", "Dallas-Fort Worth", "Phoenix"]
df['state'] = ["IL", "GA", "TX", "AZ"]
df['size'] = [10, 23, 2, 56]
df['lat'] = [41.979595, 33.64044444, 32.89595056, 33.43416667]
df['long'] = [-87.90446417, -84.42694444, -97.0372, -112.0080556]
df['text'] = "City: " + df['city'] + ", State: " + df['state']

df2 = pd.DataFrame()
df2['Title'] = ["Hospital Sketches: An Army Nurse's True Account of her Experience during the Civil War." for _ in range(100)]
df2['Author'] = [f"AuthorF AuthorN {idx}" for idx in range(100)]
df2['Date'] = [np.random.choice(list(range(1800, 1901, 1))) for _ in range(100)]


app.layout = html.Div(id="body1", children=[

    html.Div(id="header", children=[
        html.Div(id="title", children=[
            html.Div(id="t1box", children=[
                html.H1(id="title1", children=['Open'])
            ]),
            html.Div(id="t2box", children=[
                html.H1(id="title2", children=['History'])
            ])
        ]),
        html.Div(id="subtitle", children=[
            html.Div(id="st1box", children=[
                html.H2(id="subtitle1", children=['A web interface for analyzing historical literary'])
            ]),
            html.Div(id="st2box", children=[
                html.H2(id="subtitle2", children=['works from the 19th century of America'])
            ])
        ]),
    ]),



    html.Div(id="SliderBlock", children=[
        html.Div(id="sliderLabel", children=[
                html.Label('Drag the slider to adjust the desired period you are interested in:')
        ]),
        html.Div(id="slider", children=[
            dcc.RangeSlider(
                    id="sliderObj",
                    min=1800,
                    max=1900,
                    step=1,
                    marks={y: {'label': y, 'style': {'color': "black", "font-size": 15}} for y in range(1800, 1910, 10)},
                    value=[1800, 1900])
        ])
    ]),



    html.Div(id="ColumnBlock", children=[
        dcc.Graph(id="leftScatter", figure=dict(
                data=[dict(
                    type='scatter',
                )],
            layout=dict(
                    title="Most similar Documents",
                    font=dict(family='Soria, Times New Roman, Times, serif', color='#002C77', size=16),
                    margin=dict(l=50, r=50, t=50, b=50),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff"
                    )
        )),
        dcc.Graph(id="leftPie", figure=dict(
            data=[dict(
                type='bar',
            )],
            layout=dict(
                title="Most popular People",
                font=dict(family='Soria, Times New Roman, Times, serif', color='#002C77', size=16),
                margin=dict(l=20, r=0, t=50, b=20),
                legend=dict(x=0, y=1),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                orientation='h',
            )
        )
        ),


        html.Div(id="tableHeadline", children=[
            html.H4(["Collection of Books"])
        ]),
        html.Div(id="table", children=[
            html.Table([
                html.Thead([
                    html.Tr([
                            html.Th(col) for col in df2.columns
                            ])
                ]),
                html.Tbody([
                    html.Tr([
                            html.Td([
                                df2.iloc[i][col]
                            ]) for col in df2.columns
                            ]) for i in range(len(df2))
                ])
            ])
        ])
    ]),



    html.Div(id="MapBlock", children=[
        dcc.Graph(id="MapGraph", figure=dict(
                data=[dict(
                    type='scattergeo',
                    # mode='markers',
                    lon=df['long'],
                    lat=df['lat'],
                    text=df['text'],
                    hoverinfo='text',
                    marker=dict(
                        symbol='circle',
                        color="#B22234",
                        opacity=0.5,
                        size=df['size'],
                        sizemode='area',
                        sizeref=2. * max(df['size']) / (20.**2),
                        sizemin=4,
                        line=dict(width=0)
                    )
                )],
            layout=dict(
                    title='Most popular Places',
                    font=dict(family='Soria, Times New Roman, Times, serif', color='#B22234', size=16),
                    dragmode="pan",
                    geo=dict(
                        showocean=True,
                        oceancolor="#abe2fb",
                        showland=True,
                        landcolor="#AF753D",
                        showcountries=True,
                        countrywidth=0.5,
                        subunitwidth=0.5,
                        projection=dict(type="equirectangular", scale=1)
                    ),
                    margin=dict(l=0, r=0, t=30, b=30),
                    hovermode="closest",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    autosize=True,
            )
        )
        )
    ])
])


@app.callback(Output('table', 'children'),
              [Input('sliderObj', 'value')])
def update_table(value):
    relevant_data = df2[(df2.Date >= value[0]) & (df2.Date <= value[1])]
    children = [html.Table([
        html.Thead([
            html.Tr([
                html.Th(col) for col in relevant_data.columns
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td([
                    relevant_data.iloc[i][col]
                ]) for col in relevant_data.columns
            ]) for i in range(len(relevant_data))
        ])
    ])]
    return children


if __name__ == '__main__':
    app.run_server(debug=True)
