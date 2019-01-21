from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pickle
import plotly.graph_objs as go
from geopy import Nominatim
from Doc2Vec_Evaluation import get_most_similar_tokens

from app import app
from components import Header, Table, Scatter, BarOverview, Map, Dropdown, BarSpecific, Author

geolocator = Nominatim(user_agent="geolocator")

# Overview
overview_tsne = pickle.load(open("./assets/data_overview_tsne.pkl", "rb"))
overview_persons = pickle.load(open("./assets/data_overview_persons_by_tag.pkl", "rb"))
overview_places = pickle.load(open("./assets/data_overview_places_by_tag.pkl", "rb"))
# Book
author_data = pd.read_csv("./assets/data_author_information.csv", delimiter="|", index_col=0)
id_mapping = pickle.load(open("./assets/data_id_mapping.pkl", "rb"))
specific_entities = pickle.load(open("./assets/data_specific_entities_by_tag.pkl", "rb"))
doc_similarities = pickle.load(open("./assets/data_doc_similarities.pkl", "rb"))
# Word
vocabulary = list(id_mapping.keys())[308:]
remaining_persons = pickle.load(open('./assets/data_remaining_persons.pkl', 'rb'))
remaining_places = pickle.load(open('./assets/data_remaining_places.pkl', 'rb'))
cos_sim_matrix = pd.read_pickle("./assets/data_cosine_similarity_matrix.pkl")


overview = html.Div(id="body1", children=[


    Header("overview"),


    html.Div(id="ColumnBlockOverview", children=[

        Scatter(overview_tsne),

        BarOverview(overview_persons),

        html.Div(id="tableHeadline", children=[
            html.H4(["Collection of Books"])
        ]),

        Table(author_data[["Title", "Author", "Publishing Date"]]),
    ]),


    html.Div(id="MapBlock", children=[
        Map(overview_places)
    ])

])


book = html.Div(id="body1", children=[


    Header("book"),

    Dropdown("book", list(specific_entities.keys())),

    html.Div(id="ColumnBlockBook", children=[

        Author("data"),

        html.Div(id="specTitBox", children=[
            html.H1(id="specificTitle", children=[])
        ]),

        BarSpecific("DocChart"),


        BarSpecific("PersChart"),
    ]),

    html.Div(id="MapBlock", children=[
        Map(overview_places)
    ])
])


word = html.Div(id="body1", children=[


    Header("word"),

    Dropdown("word", vocabulary),

    html.Div(id="ColumnBlockWord", children=[

        html.Div(id="specTitBox", children=[
            html.H1(id="specificTitle", children=[])
        ]),

        BarSpecific("DocChart"),


        BarSpecific("PersChart"),
    ]),


    html.Div(id="MapBlock", children=[
        Map(overview_places)
    ])
])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

  if pathname == '/' or \
     pathname == '/dashboard/' or \
     pathname == '/dashboard/overview' or \
     pathname == '/dashboard/overview/':
    return overview

  if pathname == '/book' or \
          pathname == '/dashboard/book' or \
          pathname == '/dashboard/book/':
    return book

  if pathname == '/word' or \
          pathname == '/dashboard/word' or \
          pathname == '/dashboard/word/':
    return word
  else:
    return "404 Page not found"


@app.callback(Output('url', 'pathname'),
              [Input('table', 'selected_cells')])
def jump_to_book_page(cell):
  return "/dashboard/book"


@app.callback(Output('specificTitle', 'children'),
              [Input('dropdown', 'value')])
def update_specific_title(value):
  if not value:
    return ["--Select an instance from the dropdown menu--"]

  title = [value]
  return title


@app.callback(Output('AuthorBox', 'children'),
              [Input('dropdown', 'value')])
def update_author_information(value):

  data = author_data[author_data.Title == value]
  image = data["author_image"].values[0]
  if image == "-":
    image = app.get_asset_url('profile_dummy.png')
  author = data["author"].values[0].upper()
  origin = data["origin"].values[0]
  date_birth = data["date_birth"].values[0]
  birth_place = data["birth_place"].values[0]
  date_death = data["date_death"].values[0]
  occupation = data["occupation"].values[0]
  pub_date = data["Publishing Date"].values[0]
  link = data["author_wikidata_id"].values[0]
  if link == "-":
    link = None

  return [html.Div(id="AuthorImage", children=[
      html.Img(id="AImg", src=image)
  ]),
      html.Div(id="AuthorData", children=[
          html.H1("Author Information"),
          html.P(f"Name: {author}"),
          html.P(f"Origin: {origin}"),
          html.P(f"Born: {date_birth}, {birth_place}"),
          html.P(f"Date of death: {date_death}"),
          html.P(f"Occupation: {occupation}"),
          html.P(f"Publishing date of book: {pub_date}"),
          html.Br(),
          html.A("Link to Wikidata", href=link, target="_blank")
      ])
  ]


@app.callback(Output('PersChart', 'figure'),
              [Input('dropdown', 'value'),
               Input('url', 'pathname')])
def update_pers_chart(value, page):

  if not value:
    return

  if "book" in page:
    data = specific_entities[value]["persons"]
    persons = [p.title() for p in data["names"]]
    quant = data["frequency"]
    title = "<b>Most common Persons</b>"

  if "word" in page:
    persons, quant = get_most_similar_tokens(value, cos_sim_matrix, kind="persons",
                                             num=10, places=None, persons=remaining_persons)
    title = "<b>Most similar Persons</b>"

  figure = dict(
      data=[go.Bar(
          x=quant,
          y=persons,
          orientation='h',
          marker={
              'color': '#cc273c',
          },
      )],
      layout=dict(
          title=title,
          font=dict(family='Soria, Times New Roman, Times, serif', color='#002C77', size=18),
          margin=dict(l=10, r=10, t=50, b=30),
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          xaxis=dict(tick0=0, dtick=max(quant)),
          yaxis=dict(ticks='outside',
                     showgrid=True,
                     showline=False,
                     showticklabels=False),
          annotations=[dict(xref='paper', yref='y',
                            x=0, y=yd,
                            font=dict(
                                color="#000000",
                                size=14
                            ),
                            text=str(yd),
                            showarrow=False) for xd, yd in zip(quant, persons)]
      )
  )

  return figure


@app.callback(Output('DocChart', 'figure'),
              [Input('dropdown', 'value'),
               Input('url', 'pathname')])
def update_doc_chart(value, page):

  if not value:
    return

  if "book" in page:
    data = doc_similarities[value]
    books = data["books"]
    anno = [" ".join(b.split()[:(len(b.split()) // 2)]) + "<br>" + " ".join(b.split()[(len(b.split()) // 2):]) if len(b.split()) > 10 else b for idx, b in enumerate(books)]
    similarities = data["similarities"]
    title = "<b>Most similar Documents</b>"

  if "word" in page:
    books, similarities = get_most_similar_tokens(value, cos_sim_matrix, kind="docs",
                                                  num=10, places=None, persons=None)
    title = "<b>Most similar Documents</b>"

  figure = dict(
      data=[go.Bar(
          x=similarities,
          y=books,
          orientation='h',
          marker={
              'color': '#cc273c',
          },
      )],
      layout=dict(
          title=title,
          font=dict(family='Soria, Times New Roman, Times, serif', color='#002C77', size=18),
          margin=dict(l=10, r=10, t=50, b=30),
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          xaxis=dict(tick0=0, dtick=max(similarities)),
          yaxis=dict(ticks='outside',
                     showgrid=True,
                     showline=False,
                     showticklabels=False),
          annotations=[dict(xref='paper', yref='y',
                            x=0, y=yd,
                            font=dict(
                                color="#000000",
                                size=14
                            ),
                            text=str(yd),
                            showarrow=False) for xd, yd in zip(similarities, books)]
      )
  )

  return figure


@app.callback(Output('MapGraph', 'figure'),
              [Input('dropdown', 'value'),
               Input('url', 'pathname')])
def update_map(value, page):

  if not value:
    return

  pl, quant, lon, lat = [], [], [], []

  if "book" in page:
    places = specific_entities[value]["places"]["names"][-10:]
    frequency = specific_entities[value]["places"]["frequency"][-10:]
    title = "<b>Most common Places</b>"

    for idx, p in enumerate(places):
      try:
        location = geolocator.geocode(p)
        lon += [location.longitude]
        lat += [location.latitude]
        pl += [f"{p.title()}<br>Frequency: {frequency[idx]}"]
        quant += [frequency[idx]]
      except:
        pass

  if "word" in page:
    places, similarities = get_most_similar_tokens(value, cos_sim_matrix, kind="places",
                                                   num=10, places=remaining_places, persons=None)
    title = "<b>Most similar Places</b>"

    for idx, p in enumerate(places):
      try:
        location = geolocator.geocode(p)
        lon += [location.longitude]
        lat += [location.latitude]
        pl += [f"{p.title()}<br>Similarity: {similarities[idx]}"]
        quant += [similarities[idx]]
      except:
        pass

  figure = dict(
      data=[dict(
            type='scattergeo',
            lon=lon,
            lat=lat,
            text=pl,
            hoverinfo='text',
            marker=dict(
                symbol='circle',
                color="#B22234",
                opacity=0.8,
                size=quant,
                sizemode='area',
                sizeref=max(quant) / (5.**3),
                sizemin=1,
                line=dict(width=0)
            )
            )],
      layout=dict(
          title=title,
          font=dict(family='Soria, Times New Roman, Times, serif', color='#B22234', size=18),
          dragmode="pan",
          geo=dict(
              showocean=True,
              oceancolor="rgba(0, 44, 119, 0.7)",
              showland=True,
              landcolor="#ededed",  # c4c4c4, #0ba340
              lonaxis=dict(range=[min(lon) - 10, max(lon) + 10]),  # [-125, 35]
              lataxis=dict(range=[min(lat) - 10, max(lat) + 10]),  # [10, 70]
              showcountries=True,
              countrywidth=0.5,
              subunitwidth=0.5,
              projection=dict(type="equirectangular"),
          ),
          margin=dict(l=0, r=0, t=50, b=30),
          hovermode="closest",
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',
      )
  )

  return figure


if __name__ == '__main__':
  app.run_server(debug=True)
