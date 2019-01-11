from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pickle
import plotly.graph_objs as go
import math
import re

from app import app
from components import Header, Table, Scatter, BarOverview, Map, Dropdown, BarSpecific, Network


overall_data = pd.read_csv("./assets/data_overall.csv", delimiter="|", index_col=0)
overview_tsne = pickle.load(open("./assets/data_overview_tsne.pkl", "rb"))
overview_persons = pickle.load(open("./assets/data_overview_persons.pkl", "rb"))
overview_places = pickle.load(open("./assets/data_overview_places.pkl", "rb"))

mapping_title_id = pickle.load(open("./assets/data_mapping_title_id.pkl", "rb"))
specific_entities = pickle.load(open("./assets/data_specific_entities.pkl", "rb"))
doc_similarities = pickle.load(open("./assets/doc_similarities.pkl", "rb"))


overview = html.Div(id="body1", children=[


    Header("overview"),


    html.Div(id="ColumnBlockOverview", children=[

        Scatter(overview_tsne),

        BarOverview(overview_persons),

        html.Div(id="tableHeadline", children=[
            html.H4(["Collection of Books"])
        ]),

        Table(overall_data[["Title", "Author", "Publishing Date"]]),
    ]),


    html.Div(id="MapBlock", children=[
        Map(overview_places)
    ])

])


specific = html.Div(id="body1", children=[


    Header("specific"),

    Dropdown(list(specific_entities.keys())),

    html.Div(id="ColumnBlockSpecific", children=[

        html.Div(id="bTbox", children=[
            html.H1(id="bookTitle", children=['Book Title'])
        ]),

        Network(),


        BarSpecific(),
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

  if pathname == '/specific' or \
          pathname == '/dashboard/specific' or \
          pathname == '/dashboard/specific/':
    return specific
  else:
    return "404 Page not found"


@app.callback(Output('bookTitle', 'children'),
              [Input('dropdown', 'value')])
def update_title(value):
  book = [value]

  return book


@app.callback(Output('network', 'figure'),
              [Input('dropdown', 'value')])
def update_network(value):

  book_of_interest = value
  books = doc_similarities[value]["books"][-5:]
  similarities = doc_similarities[value]["similarities"][-5:]

  hover_text = [" ".join(b.split()[:(len(b.split()) // 2)]) + "<br>" + " ".join(b.split()[(len(b.split()) // 2):]) + "<br>Similarity: " + str(similarities[idx]) if len(b.split()) > 10 else b + "<br>Similarity: " + str(similarities[idx]) for idx, b in enumerate(books)]

  nodes = [(0, -1),
           (0.9511, -0.309),
           (0.5878, 0.809),
           (-0.5878, 0.809),
           (-0.9511, -0.309)]
  nodes = [(nodes[idx][0] * (1 - s * s) * 10, nodes[idx][1] * (1 - s * s) * 10)
           for idx, s in enumerate(similarities)]
  nodes = [(0, 0)] + nodes

  edges = {"x": [], "y": []}
  for idx, n in enumerate(nodes[1:]):
    if idx == 0:
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
      text=[None] + hover_text,
      mode='markers',
      hoverinfo='text',
      marker=dict(
          symbol='circle',
          color="#002C77",
          sizemode='area',
          size=50,
      )
  )

  figure = dict(
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
      )
  )

  return figure


@app.callback(Output('BarSpecific', 'figure'),
              [Input('dropdown', 'value')])
def update_bar(value):
  try:
    data = specific_entities[value]["persons"]
  except:
    return
  persons = list(data.keys())
  frequency = list(data.values())

  figure = dict(
      data=[go.Bar(
          x=frequency,
          y=persons,
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
          xaxis=dict(tick0=0, dtick=max(frequency)),
          yaxis=dict(ticks='outside'),
      )
  )

  return figure


if __name__ == '__main__':
  app.run_server(debug=True)
