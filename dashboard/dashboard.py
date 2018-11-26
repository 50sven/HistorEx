import dash
import dash_core_components as dcc
import dash_html_components as html
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

scl = [[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"],
       [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]]

df2 = pd.DataFrame()
df2['Title'] = ["Hospital Sketches: An Army Nurse's True Account of her Experience during the Civil War." for _ in range(100)]
df2['Author'] = [f"AuthorF AuthorN {idx}" for idx in range(100)]
df2['Date'] = [1999 for _ in range(100)]


app.layout = html.Div(id="body1", children=[

    html.Div(id="Headline", children=[
        html.H1('OPEN HISTORY'),
        html.H5('A web interface for analyzing historical literary works from the 19th century of America')
    ]),
    html.Div(id="SliderBlock", children=[
        html.Div(id="sliderLabel", children=[
            html.Label('Drag the slider to adjust the desired period you are interested in:')
        ]),
        html.Div(id="slider", children=[
            dcc.RangeSlider(
                min=1800,
                max=1900,
                step=1,
                marks={y: y for y in range(1800, 1910, 10)},
                value=[1800, 1900])
        ])
    ]),
    html.Div(id="ColumnBlock", children=[
        dcc.Graph(id="left_scatter", figure=dict(
                data=[dict(
                    type='scatter',
                )],
            layout=dict(
                    title="Scatter Plot",
                    margin=dict(l=50, r=50, t=50, b=50),
            )
        )),
        dcc.Graph(id="left_pie", figure=dict(
            data=[dict(
                values=[10, 90],
                type='pie'
            )],
            layout=dict(
                title="Pie Chart",
                margin=dict(l=0, r=0, t=30, b=30),
                legend=dict(x=0, y=1),
            )
        )
        ),
        html.Div(id="table_headline", children=[
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
        dcc.Graph(id="Map_Graph", figure=dict(
                data=[dict(
                    type='scattergeo',
                    # mode='markers',
                    lon=df['long'],
                    lat=df['lat'],
                    text=df['text'],
                    hoverinfo='text',
                    marker=dict(
                        symbol='circle',
                        color="red",
                        opacity=0.5,
                        size=df['size'],
                        sizemode='area',
                        sizeref=2. * max(df['size']) / (20.**2),
                        sizemin=4,
                        line=dict(width=0)
                    )
                )],
            layout=dict(
                    title='My Map',
                    dragmode="pan",
                    geo=dict(
                        showocean=True,
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


if __name__ == '__main__':
    app.run_server(debug=True)
