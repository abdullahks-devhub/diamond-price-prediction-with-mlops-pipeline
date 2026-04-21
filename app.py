import pickle
import pandas as pd
import gradio as gr
import numpy as np

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

def predict(carat, cut, color, clarity, depth, table, x, y, z):
    try:
        if not all([carat, cut, color, clarity, depth, table, x, y, z]):
            return "⚠️ Please fill in all fields"

        data = {
            'carat': [carat], 'cut': [cut], 'color': [color],
            'clarity': [clarity], 'depth': [depth], 'table': [table],
            'x': [x], 'y': [y], 'z': [z]
        }
        df = pd.DataFrame(data)
        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)
        price = float(prediction[0])

        if price < 1000:
            tier = "💎 Entry Level"
        elif price < 5000:
            tier = "💎💎 Mid Range"
        elif price < 15000:
            tier = "💎💎💎 Premium"
        else:
            tier = "💎💎💎💎 Luxury"

        return f"${price:,.2f}\n{tier}"

    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft(), title="Diamond Price Predictor") as interface:

    gr.Markdown("""
    # 💎 Diamond Price Predictor
    ### Predict the market price of any diamond instantly
    Enter the diamond's characteristics below to get an accurate price estimate.
    ---
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚖️ Physical Properties")
            carat = gr.Number(label="Carat Weight", value=0.5,
                            info="Weight of the diamond (0.2 - 5.0)")
            depth = gr.Number(label="Depth %", value=61.5,
                            info="Total depth percentage")
            table = gr.Number(label="Table %", value=55.0,
                            info="Width of top of diamond relative to widest point")

        with gr.Column():
            gr.Markdown("### 📐 Dimensions (mm)")
            x = gr.Number(label="Length (x)", value=3.95)
            y = gr.Number(label="Width (y)", value=3.98)
            z = gr.Number(label="Height (z)", value=2.43)

        with gr.Column():
            gr.Markdown("### ✨ Quality Grades (4 C's)")
            cut = gr.Dropdown(
                ["Fair", "Good", "Very Good", "Premium", "Ideal"],
                label="Cut", value="Ideal",
                info="Quality of the cut"
            )
            color = gr.Dropdown(
                ["D", "E", "F", "G", "H", "I", "J"],
                label="Color", value="E",
                info="D=Colorless (best), J=Light Yellow"
            )
            clarity = gr.Dropdown(
                ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
                label="Clarity", value="SI1",
                info="IF=Internally Flawless (best), I1=Inclusions"
            )

    with gr.Row():
        predict_btn = gr.Button("💎 Predict Price", variant="primary", size="lg")
        clear_btn = gr.Button("🔄 Clear", size="lg")

    with gr.Row():
        output = gr.Textbox(
            label="Predicted Price",
            lines=2,
            show_copy_button=True
        )

    gr.Markdown("""
    ---
    ### 📊 Quick Reference
    | Grade | Cut | Color | Clarity |
    |-------|-----|-------|---------|
    | Best | Ideal | D | IF |
    | Great | Premium | E-F | VVS1-VVS2 |
    | Good | Very Good | G-H | VS1-VS2 |
    | Fair | Good/Fair | I-J | SI1-SI2 |

    > ⚠️ *Predictions are based on the Kaggle Diamonds Dataset and are for educational purposes only.*
    """)

    predict_btn.click(
        fn=predict,
        inputs=[carat, cut, color, clarity, depth, table, x, y, z],
        outputs=output
    )
    clear_btn.click(
        fn=lambda: [0.5, "Ideal", "E", "SI1", 61.5, 55.0, 3.95, 3.98, 2.43, ""],
        outputs=[carat, cut, color, clarity, depth, table, x, y, z, output]
    )

    gr.Examples(
        examples=[
            [0.23, "Ideal", "E", "SI2", 61.5, 55.0, 3.95, 3.98, 2.43],
            [1.01, "Premium", "G", "SI1", 62.5, 57.0, 6.35, 6.39, 3.98],
            [2.5, "Very Good", "D", "VS1", 60.0, 58.0, 8.72, 8.68, 5.22],
        ],
        inputs=[carat, cut, color, clarity, depth, table, x, y, z],
        label="📌 Try These Examples"
    )

interface.launch(debug=True)