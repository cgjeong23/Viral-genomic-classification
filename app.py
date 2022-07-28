# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd

from ML.inference import infer, load_for_inference

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

model_path = 'ML/model.pth'
model, tokenizer, label_dict = load_for_inference(model_path, "ML/gene_tokenizer.json", "ML/label_dict.json",
embedding_dim=256, hidden_dim=64, num_layers=1)

df = pd.DataFrame({
    "Virus Type": [x[0] for x in sorted(label_dict.items(), key=lambda x: x[1])],
})

app.layout = html.Div([
    html.H1("Viral Genome Classification"),
    html.Hr(),
    html.Div([
        html.H6("Upload your RNA Sequence."),
        html.H6("We will detect the virus for you."),
    ]),
    html.Hr(),
    "Type your RNA Sequence: ",
    html.Div(dcc.Textarea(id='my-input', value='', placeholder="ACTG...")),
    html.Button("Go!", id='go-button'),
    html.Br(),
    html.Div(id='my-output'),
], style={'text-align':'center'})


@app.callback(Output(component_id='my-output', component_property='children'),
              State(component_id='my-input', component_property='value'),
              Input(component_id='go-button', component_property='n_clicks'))

def update_output_div(input_value, n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    input_value = input_value.strip().upper()

    if set(input_value) != set('ACTG'):
        return html.P("Please input valid sequence containing only A, C, T, G", style={'color':'red'})

    probability = infer(input_value, tokenizer, model)
    df['Probability'] = probability
    fig = px.bar(df, x="Virus Type", y="Probability", color="Virus Type", title="Classification Result")
    return (dcc.Graph(id='example-graph', figure=fig))


if __name__ == '__main__':
    app.run_server()
