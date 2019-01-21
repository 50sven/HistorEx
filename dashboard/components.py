import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import dash_table
from app import app


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
        html.Button("BOOK", id="btn2", className="btn")
    ], href="/dashboard/book")

    button3 = html.A(id="btn3box", children=[
        html.Button("WORD", id="btn3", className="btn")
    ], href="/dashboard/word")

    if page == "overview":
        button1.children[0].style = style
    if page == "book":
        button2.children[0].style = style
    if page == "word":
        button3.children[0].style = style

    return html.Div(id="header", children=[
        get_title(),
        get_subtitle(),
        button1,
        button2,
        button3
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
                color='#B22234',
            ),
            textfont=dict(family='Soria, Times New Roman, Times, serif')
        )],
        layout=dict(
            title="<b>Document Similarities</b>",
            hovermode='closest',
            font=dict(family='Soria, Times New Roman, Times, serif', color='#002C77', size=18),
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
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
            x=data["frequencies"],
            y=data["names"],
            orientation='h',
            marker={
                'color': '#e02b42',
            },
        )],
        layout=dict(
            title="<b>Most common Persons</b>",
            font=dict(family='Soria, Times New Roman, Times, serif', color='#002C77', size=18),
            margin=dict(l=10, r=20, t=50, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tick0=0, dtick=5000),
            yaxis=dict(ticks='outside',
                       showgrid=True,
                       showline=False,
                       showticklabels=False),
            annotations=[dict(xref='paper', yref='y',
                              x=0, y=yd,
                              font=dict(
                                  color="#000000",
                                  size=15
                              ),
                              text=str(yd),
                              showarrow=False) for xd, yd in zip(data["frequencies"], data["names"])]
        )
    ))


def Table(data):
    """Returns Table for overview page
    """
    data.columns = ["Title", "Author", "Date"]
    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict("rows"),
        style_as_list_view=True,
        sorting=True,
        style_header={
            'fontWeight': 'bold'
        },
        style_cell={
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Title'},
             'textAlign': 'left'},
            {'if': {'column_id': 'Author'},
             'textAlign': 'left'},
            {'if': {'column_id': 'Date'},
             'textAlign': 'center'},
        ],
        style_data={'whiteSpace': 'normal'},
    )


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
                size=data['frequencies'],
                sizemode='area',
                sizeref=max(data['frequencies']) / (5.**3),
                sizemin=1,
                line=dict(width=0)
            )
        )],
        layout=dict(
            title='<b>Most common Places</b>',
            font=dict(family='Soria, Times New Roman, Times, serif', color='#B22234', size=18),
            dragmode="pan",
            geo=dict(
                showocean=True,
                oceancolor="rgba(0, 44, 119, 0.7)",
                showland=True,
                landcolor="#ededed",  # c4c4c4, #0ba340
                lonaxis=dict(range=[min(lon) - 10, max(lon) + 10]),
                lataxis=dict(range=[min(lat) - 10, max(lat) + 10]),
                showcountries=True,
                countrywidth=0.5,
                subunitwidth=0.5,
                projection=dict(type="equirectangular")
            ),
            margin=dict(l=0, r=0, t=50, b=30),
            hovermode="closest",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True,
        )
    ))


def Dropdown(page, data):
    """Returns Dropdown Menu for specific page
    """
    if page == "book":
        ph = "Select a book from the collection"
    if page == "word":
        ph = "Select a word from the vocabulary"

    return html.Div(id="ddbox", children=[
        dcc.Dropdown(
            id='dropdown',
            options=[{"label": x, "value": x} for x in data],
            placeholder=ph,
        )
    ])


def BarSpecific(id_tag):
    """Returns Bar Chart for overview persons or specific document similarity
    """
    if id_tag == "DocChart":
        data = {'persons': ["Documents"], 'frequency': [1]}
        title = "<b>Document Similarities</b>"
    if id_tag == "PersChart":
        data = {'persons': ["Persons"], 'frequency': [1]}
        title = "<b>Common Persons</b>"

    return dcc.Graph(id=id_tag, className="bar", figure=dict(
        data=[go.Bar(
            x=data["frequency"],
            y=data["persons"],
            orientation='h',
            marker={
                'color': '#e02b42',
            },
        )],
        layout=dict(
            title=title,
            font=dict(family='Soria, Times New Roman, Times, serif', color='#002C77', size=18),
            margin=dict(l=100, r=20, t=50, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tick0=0, dtick=1),
            yaxis=dict(ticks='outside')
        )
    ))


def Author(data):
    """Returns Author information for a specific book
    """
    return html.Div(id="AuthorBox", children=[
        html.Div(id="AuthorImage", children=[
            html.Img(id="AImg", src=app.get_asset_url('profile_dummy.png'))
        ]),
        html.Div(id="AuthorData", children=[
            html.H1("Author Information"),
            html.P("Name: name"),
            html.P("Born: date, place"),
            html.P("Date of death: date"),
            html.P("Origin: place"),
            html.P("Occupation: occupation"),
            html.P("Publishing date of book: date"),
            html.Br(),
            html.A("Link to Wikidata", href='http://www.google.com', target="_blank")
        ])
    ])
