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


@app.callback(Output('network', 'figure'),
              [Input('dropdown', 'value')])
def update_network(value):

  if not value:
    return

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
