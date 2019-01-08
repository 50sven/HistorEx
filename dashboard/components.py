import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import math


def Header(page):
    """Returns header for end points 'overview' and 'specific'
    """
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


def get_title():
    """Subfunction for Header
    """
    return html.Div(id="title", children=[
        html.Div(id="t1box", children=[
            html.H1(id="title1", children=['Open'])
        ]),
        html.Div(id="t2box", children=[
            html.H1(id="title2", children=['History'])
        ])
    ])


def get_subtitle():
    """Subfunction for Header
    """
    return html.Div(id="subtitle", children=[
        html.Div(id="st1box", children=[
            html.H2(id="subtitle1", children=['A web interface for analyzing historical literary'])
        ]),
        html.Div(id="st2box", children=[
            html.H2(id="subtitle2", children=['works from the 19th century of America'])
        ])
    ])


def Scatter(data):
    """Returns Scatter Plot for global document similarity
    """
    return dcc.Graph(id="leftScatter", figure=dict(
        data=[go.Scatter(
            x=data["x"],
            y=data["y"],
            text=data["titles"],
            hoverinfo='text',
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.5,
                color='#002C77',
            ),
            textfont=dict(family='Soria, Times New Roman, Times, serif')
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


def BarOverview(data):
    """Returns Bar Chart for overview persons or specific document similarity
    """
    return dcc.Graph(id="BarOverview", className="bar", figure=dict(
        data=[go.Bar(
            x=data["occurence"],
            y=data["names"],
            orientation='h',
            marker={
                'color': '#002C77',
            },
        )],
        layout=dict(
            title="Most popular People",
            font=dict(family='Soria, Times New Roman, Times, serif', color='#B22234', size=18),
            margin=dict(l=110, r=20, t=50, b=30),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(tick0=0, dtick=5000),
            yaxis=dict(ticks='outside')
        )
    ))


def Table(data):
    """Returns Table for overview page
    """
    return html.Div(id="table", children=[
        html.Table([
            html.Thead([
                    html.Tr([
                            html.Th(col) for col in ["Title", "Author", "Date"]
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


def Map(data):
    """Returns Map for places in both endpoints
    """
    lon = [loc[1] for loc in data["geo"]]
    lat = [loc[0] for loc in data["geo"]]

    return dcc.Graph(id="MapGraph", figure=dict(
        data=[dict(
            type='scattergeo',
            # mode='markers',
            lon=lon,
            lat=lat,
            text=data["names"],
            hoverinfo='text',
            marker=dict(
                symbol='circle',
                color="#B22234",
                opacity=0.8,
                size=data['occurence'],
                sizemode='area',
                sizeref=max(data['occurence']) / (5.**3),
                sizemin=1,
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
                lonaxis=dict(range=[-125, 35]),
                lataxis=dict(range=[10, 70]),
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


def Dropdown(data):
    """Returns Dropdown Menu for specific page
    """
    return html.Div(id="ddbox", children=[
        dcc.Dropdown(
            id='dropdown',
            options=[{"label": x, "value": x} for x in data],
            placeholder="Select the book of interest",
            style={
            }
        )
    ])


def BarSpecific():
    """Returns Bar Chart for overview persons or specific document similarity
    """
    data = {'persons': ["persons"], 'frequency': [1]}
    return dcc.Graph(id="BarSpecific", className="bar", figure=dict(
        data=[go.Bar(
            x=data["frequency"],
            y=data["persons"],
            orientation='h',
            marker={
                'color': '#002C77',
            },
        )],
        layout=dict(
            title="Most popular Persons",
            font=dict(family='Soria, Times New Roman, Times, serif', color='#B22234', size=18),
            margin=dict(l=110, r=20, t=50, b=30),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(tick0=0, dtick=1),
            yaxis=dict(ticks='outside')
        )
    ))


def Network():
    """Returns similarity network for a specific book
    """
    book_of_interest = "Book0"
    similarities = {"book2": 0.9, "book3": 0.5, "book4": 0.7, "book5": 0.1, "book6": 0.3}

    nodes = [(0, -1),
             (0.9511, -0.309),
             (0.5878, 0.809),
             (-0.5878, 0.809),
             (-0.9511, -0.309)]
    nodes = [(nodes[idx][0] * math.sqrt((1 - v) * 10), nodes[idx][1] * math.sqrt((1 - v) * 10)) for idx, (b, v) in enumerate(similarities.items())]
    nodes.append((0, 0))

    edges = {"x": [], "y": []}
    for idx, n in enumerate(nodes[:-1]):
        if idx == len(nodes):
            pass
        edges["x"] += (0, n[0], None)
        edges["y"] += (0, n[1], None)

    edge_trace = go.Scatter(
        x=edges["x"],
        y=edges["y"],
        line=dict(width=5, color='#B22234'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=[n[0] for n in nodes],
        y=[n[1] for n in nodes],
        text=[b for b in similarities.keys()] + [None],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            symbol='circle',
            color="#002C77",
            sizemode='area',
            size=50,
        ))

    graph = dcc.Graph(id="network", figure=dict(
        data=[edge_trace, node_trace],
        layout=dict(
            title="Most similar Documents",
            font=dict(family='Soria, Times New Roman, Times, serif',
                      color='#B22234',
                      size=18),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(showgrid=False,
                       showline=False,
                       showticklabels=False,
                       zeroline=False),
            yaxis=dict(showgrid=False,
                       showline=False,
                       showticklabels=False,
                       zeroline=False)
        ))
    )

    return graph


def Author(data):
    """Returns Author information for a specific book
    """
    pass


def List(data):
    """Returns List of most popular persons for a spcecific book
    """
    pass
