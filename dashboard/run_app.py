from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pickle

from app import app
from components import Header, Table, Scatter, Bar, Map

# overview_tsne = pickle.load(open("TSNE.pkl", "rb"))
# overview_tsne = ([x[0] for x in overview_tsne],
#                  [x[1][0] for x in overview_tsne],
#                  [x[1][1] for x in overview_tsne])
overview_tsne = pickle.load(open("./assets/overview_tsne.pkl", "rb"))

# overview_place_person = pickle.load(open("overview_entities.pkl", "rb"))
# overview_places, overview_persons = zip(*overview_place_person)
# overview_places, overview_persons = (list(overview_places),
#                                      list(overview_persons))
# overview_places, overview_persons = (sorted(overview_places, key=lambda tup: (tup[1], tup[0])),
#                                      sorted(overview_persons, key=lambda tup: (tup[1], tup[0])))s
overview_persons = pickle.load(open("./assets/overview_persons.pkl", "rb"))
overview_places = pickle.load(open("./assets/overview_places.pkl", "rb"))

overview_persons = ([x[0] for x in overview_persons[-10:]],
                    [x[1] for x in overview_persons[-10:]])
overview_places = ([x[0] for x in overview_places],
                   [x[1] for x in overview_places])

overview_table = pd.read_csv("./assets/overview_table.csv", index_col=0)

map_data = pd.read_pickle("random_places.pkl")


overview = html.Div(id="body1", children=[


    Header("overview"),


    html.Div(id="ColumnBlock", children=[

        Scatter(overview_tsne),

        Bar(overview_persons),

        html.Div(id="tableHeadline", children=[
            html.H4(["Collection of Books"])
        ]),

        Table(overview_table),
    ]),


    html.Div(id="MapBlock", children=[
        Map(map_data)
    ])

])


specific = html.Div(id="body1", children=[


    Header("specific"),


    html.Div(id="ColumnBlock", children=[

        Scatter(overview_tsne),

        Bar(overview_persons),

        html.Div(id="tableHeadline", children=[
            html.H4(["Collection of Books"])
        ]),

        Table(overview_table),
    ]),


    html.Div(id="MapBlock", children=[
        Map(map_data)
    ])
])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
  if pathname == '/dashboard' or \
     pathname == '/dashboard/' or \
     pathname == '/dashboard/overview' or \
     pathname == '/dashboard/overview/':
    return overview
  elif pathname == '/dashboard/specific' or \
          pathname == '/dashboard/specific/':
    return specific
  else:
    return "404 Page not found"


if __name__ == '__main__':
  app.run_server(debug=True)
