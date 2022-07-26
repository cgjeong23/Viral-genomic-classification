# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd

app = Dash(__name__)

df = pd.DataFrame({
    "Virus Type": [
        "Coronaviridae", "Influenza", "Metapneumovirus", "Rhinovirus", "SARS-CoV-2",
        "Human (no virus)"
    ],
    "Probability": [0.8, 0.1, 0, 0, 0.1, 0],
})

app.layout = html.Div([
    html.H1("Viral Genome Classification"),
    html.Hr(),
    html.Div([
        html.H6("Upload your RNA Sequence."),
        html.H6("We will detect the virus for you."),
    ]),
    html.Hr(),
    "Input: ",
    html.Div(dcc.Textarea(id='my-input', value='', placeholder="input value")),
    html.Button("Go!", id='go-button'),
    html.Br(),
    html.Div(id='my-output'),
])


@app.callback(Output(component_id='my-output', component_property='children'),
              State(component_id='my-input', component_property='value'),
              Input(component_id='go-button', component_property='n_clicks'))
def update_output_div(input_value, n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    # city_df = df[df['City'] == input_value]
    fig = px.bar(df, x="Virus Type", y="Probability", color="Virus Type")
    return dcc.Graph(id='example-graph', figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
