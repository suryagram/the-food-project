import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go

import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def read_data_and_preprocess():
    df = pd.read_csv("../data/the-food-project/clean_recipes.csv",sep=';')
    df['Review Count'] = (df['Review Count'].replace(r'[km]+$', '', regex=True).astype(float) * \
          df['Review Count'].str.extract(r'[\d\.]+([km]+)', expand=False)
             .fillna(1)
             .replace(['k','m'], [10**3, 10**6]).astype(int))
    df = df.loc[df['Review Count'] > 1200]
    return df

def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))

def createGraph(df):
    G = nx.Graph()
    for index, row in df.iterrows():
        ingredients = row['Ingredients']
        ingredients_list = ingredients.split(',')
        G.add_nodes_from(ingredients_list)
        edge_tuples = rSubset(ingredients_list, 2)
        G.add_edges_from(edge_tuples)

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    for n, p in pos.items():
        G.nodes[n]['pos'] = p
    return G

def create_edge_and_node_trace(G):
    print(G.number_of_edges())
    print(G.number_of_nodes())
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
    i = 0
    for edge in G.edges():
        i = i + 1
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))
    i = 0
    for node in G.nodes():
        i = i + 1
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])
    return node_trace, edge_trace

def generate_graph(node_trace, edge_trace):
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Food Recipe Connections',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="No. of connections",
                        showarrow=False,
                        xref="paper", yref="paper") ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig


df = read_data_and_preprocess()
G = createGraph(df)
edge_trace, node_trace = create_edge_and_node_trace(G)
fig = generate_graph(edge_trace, node_trace)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(children=[
    html.H1(children='The Food Project'),

    html.Div(children='''
        Visualizing Ingredients, and Recipes for better Food decisions
    '''),

    html.Div(dcc.Graph(id='Graph', figure=fig)),
    html.Div(className='row', children=[
                    html.Div([html.H2('Overall Data'),
                              html.P('Num of nodes: ' + str(len(G.nodes))),
                              html.P('Num of edges: ' + str(len(G.edges)))],
                              className='three columns'),
                    html.Div([
                            html.H2('Selected Data'),
                            html.Div(id='selected-data'),
                        ], className='six columns')
                    ])

])

@app.callback(
    Output('selected-data', 'children'),
    [Input('Graph','selectedData')])
def display_selected_data(selectedData):
    num_of_nodes = len(selectedData['points'])
    text = [html.P('Num of nodes selected: '+str(num_of_nodes))]
    for x in selectedData['points']:
        material = int(x['text'].split('<br>')[0][10:])
        text.append(html.P(str(material)))
    return text







if __name__ == '__main__':
    app.run_server(debug=True)
