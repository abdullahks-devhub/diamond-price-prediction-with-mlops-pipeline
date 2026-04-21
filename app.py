import pickle
import pandas as pd
import gradio as gr

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

def predict(carat, cut, color, clarity, depth, table, x, y, z):
    try:
        data = {
            'carat': carat,
            'cut': cut,
            'color': color,
            'clarity': clarity,
            'depth': depth,
            'table': table,
            'x': x,
            'y': y,
            'z': z
        }

        df = pd.DataFrame(data)
        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)

        return float(prediction[0])
    except Exception as e:
        return str(e)

interface = gr.Interface(
    fn = predict,
    inputs=[
        gr.Number(label='Carat'),
        gr.Dropdown(["Fair", "Good", "Very Good", "Premium", "Ideal"], label="Cut"),
        gr.Dropdown(["D", "E", "F", "G", "H", "I", "J"], label="Color"),
        gr.Dropdown(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], label="Clarity"),
        gr.Number(label="Depth"),
        gr.Number(label="Table"),
        gr.Number(label="x"),
        gr.Number(label="y"),
        gr.Number(label="z")
    ],
    outputs="number",
    title="Diamond Price Prediction",
    description="Enter Diamond Features to predict price"
)

interface.launch(debug=True)