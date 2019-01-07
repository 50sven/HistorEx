import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go


#### Create global header for end points ####
def Header(page):
    style = {
        "background-color": "var(--red_color)",
        "color": "black",
        "text-shadow": "0 0 10px #ffffff"
    }

    button1 = html.A(id="btn1box", children=[
        html.Button("OVERVIEW", id="btn1", className="btn")
    ], href="/dashboard/overview")

    button2 = html.A(id="btn2box", children=[
        html.Button("SPECIFIC", id="btn2", className="btn")
    ], href="/dashboard/specific")

    if page == "overview":
        button1.children[0].style = style
    if page == "specific":
        button2.children[0].style = style

    return html.Div(id="header", children=[
        get_title(),
        get_subtitle(),
        button1,
        button2
    ])


# Subfunction for header
def get_title():
    return html.Div(id="title", children=[
        html.Div(id="t1box", children=[
            html.H1(id="title1", children=['Open'])
        ]),
        html.Div(id="t2box", children=[
            html.H1(id="title2", children=['History'])
        ])
    ])


# Subfunction for header
def get_subtitle():
    return html.Div(id="subtitle", children=[
        html.Div(id="st1box", children=[
            html.H2(id="subtitle1", children=['A web interface for analyzing historical literary'])
        ]),
        html.Div(id="st2box", children=[
            html.H2(id="subtitle2", children=['works from the 19th century of America'])
        ])
    ])


# Subfunction for graphs
def Scatter(data):
    return dcc.Graph(id="leftScatter", figure=dict(
        data=[go.Scatter(
            x=data[1],
            y=data[2],
            text=data[0],
            hoverinfo='text',
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.5,
                'color': '#002C77'
            },
        )],
        layout=dict(
            title="Most similar Documents",
            hovermode='closest',
            font=dict(family='Soria, Times New Roman, Times, serif', color='#B22234', size=18),
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(showgrid=True,
                       showline=False,
                       showticklabels=False),
            yaxis=dict(showgrid=True,
                       showline=False,
                       showticklabels=False)
        )
    ))


# Subfunction for graphs
def Bar(data=None):
    return dcc.Graph(id="leftBar", figure=dict(
        data=[go.Bar(
            x=data[1],
            y=data[0],
            orientation='h',
            marker={
                'color': '#002C77',
            },
        )],
        layout=dict(
            title="Most popular People",
            font=dict(family='Soria, Times New Roman, Times, serif', color='#B22234', size=18),
            margin=dict(l=90, r=10, t=50, b=30),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(tick0=0, dtick=5000),
            yaxis=dict(ticks='outside')
        )
    ))


#### Creates table for overview ####
def Table(data=None):
    return html.Div(id="table", children=[
        html.Table([
            html.Thead([
                    html.Tr([
                            html.Th(col) for col in data.columns
                            ])
                    ]),
            html.Tbody([
                html.Tr([
                    html.Td([
                        data.iloc[i][col]
                    ]) for col in data.columns
                ]) for i in range(len(data))
            ])
        ])
    ])


#### Creates Map for overview ####
def Map(data=None):
    return dcc.Graph(id="MapGraph", figure=dict(
        data=[dict(
            type='scattergeo',
            # mode='markers',
            lon=data['long'],
            lat=data['lat'],
            text=data['text'],
            hoverinfo='text',
            marker=dict(
                symbol='circle',
                color="#B22234",
                opacity=0.8,
                size=data['size'],
                sizemode='area',
                sizeref=max(data['size']) / (20.**2),
                sizemin=5,
                line=dict(width=0)
            )
        )],
        layout=dict(
            title='Most popular Places',
            font=dict(family='Soria, Times New Roman, Times, serif', color='#B22234', size=18),
            dragmode="pan",
            geo=dict(
                showocean=True,
                oceancolor="rgba(0, 44, 119, 0.7)",
                showland=True,
                landcolor="#ededed",  # c4c4c4, #0ba340
                showcountries=True,
                countrywidth=0.5,
                subunitwidth=0.5,
                projection=dict(type="equirectangular", scale=1)
            ),
            margin=dict(l=0, r=0, t=50, b=30),
            hovermode="closest",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True,
        )
    ))
