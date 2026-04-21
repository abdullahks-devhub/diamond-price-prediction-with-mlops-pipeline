import os
import pickle
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download
from functools import lru_cache

# Constants
REPO_ID = "AbdullahKS-Devhub/diamond-price-model"


@lru_cache()
def load_artifacts():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pkl")
        preprocessor_path = hf_hub_download(repo_id=REPO_ID, filename="preprocessor.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        return model, preprocessor
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None


model, preprocessor = load_artifacts()


def predict(carat, cut, color, clarity, depth, table, x, y, z):
    if model is None or preprocessor is None:
        return "❌ Error: Artifacts unavailable", "Error"

    try:
        inputs = [carat, cut, color, clarity, depth, table, x, y, z]
        if any(v is None for v in inputs):
            return "⚠️ Fill all fields", "Pending"

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

        # Format price beautifully
        formatted_price = f"${price:,.2f}"

        return formatted_price, tier

    except Exception as e:
        return "Error", str(e)


# Premium CSS Styling
premium_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Global Fonts & Background */
.gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #09090e 0%, #1a1a2e 100%) !important;
    color: #e0e0e0 !important;
}

/* Glassmorphism Cards */
.form, .block, .tabs, [class*='gradio-accordion'], .contain {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
}

/* Hide some default Gradio borders */
.svelte-1g7k1ee, .svelte-1gfknrx {
    border: none !important;
}

/* Typography Enhancements */
h1, h2, h3 {
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(139, 92, 246, 0.4);
    letter-spacing: 0.5px;
}

/* Custom Buttons */
button.primary {
    background: linear-gradient(45deg, #6b21a8, #3b82f6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(107, 33, 168, 0.5) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6) !important;
}

button.secondary {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #fff !important;
    transition: all 0.3s ease !important;
}
button.secondary:hover {
    background: rgba(255,255,255,0.1) !important;
}

/* Inputs styling */
input, select {
    background: rgba(0, 0, 0, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #fff !important;
    border-radius: 8px !important;
}
input:focus, select:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3) !important;
}

/* Hero Section Image */
#hero-image img {
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
}

/* Prediction Output Area */
#prediction-box {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(88, 28, 135, 0.15) 0%, rgba(30, 58, 138, 0.15) 100%) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 16px;
    box-shadow: inset 0 0 20px rgba(139, 92, 246, 0.1);
}
#price-text textarea {
    font-size: 3rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    background: -webkit-linear-gradient(45deg, #c084fc, #60a5fa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
"""

theme = gr.themes.Base(
    primary_hue="purple",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "system-ui", "sans-serif"]
)

with gr.Blocks(theme=theme, css=premium_css, title="Diamond Appraiser") as interface:
    # Hero Section
    with gr.Row():
        with gr.Column(scale=1):
            if os.path.exists("hero.png"):
                gr.Image("hero.png", show_label=False, show_download_button=False, interactive=False, container=False,
                         elem_id="hero-image")
        with gr.Column(scale=2):
            gr.Markdown("""
            # 💎 Precision Diamond Appraisal
            ### *AI-Powered Market Valuation*

            Welcome to the future of diamond pricing. Enter the precise physical dimensions and GIA quality grades of your diamond to receive an instant, accurate market valuation powered by our ensemble ML pipeline.
            """)

    gr.Markdown("---")

    with gr.Row():
        # Input Section (Left)
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("✨ 4 C's (Quality)"):
                    carat = gr.Slider(minimum=0.2, maximum=5.0, value=0.5, step=0.01, label="Carat Weight",
                                      info="Size and weight of the diamond")
                    with gr.Row():
                        cut = gr.Dropdown(["Fair", "Good", "Very Good", "Premium", "Ideal"], label="Cut Grade",
                                          value="Ideal")
                        color = gr.Dropdown(["D", "E", "F", "G", "H", "I", "J"], label="Color Grade", value="E",
                                            info="D (Best) to J (Warm)")
                        clarity = gr.Dropdown(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"], label="Clarity",
                                              value="VS1", info="IF (Flawless) to I1 (Included)")

                with gr.TabItem("📐 Dimensions"):
                    with gr.Row():
                        x = gr.Number(label="Length (x mm)", value=5.1)
                        y = gr.Number(label="Width (y mm)", value=5.12)
                        z = gr.Number(label="Height (z mm)", value=3.15)
                    with gr.Row():
                        depth = gr.Number(label="Depth %", value=61.5, info="Total depth percentage")
                        table = gr.Number(label="Table %", value=55.0, info="Width of top relative to widest point")

            with gr.Row():
                clear_btn = gr.Button("Reset Fields", variant="secondary")
                predict_btn = gr.Button("Calculate Valuation", variant="primary")

            gr.Examples(
                examples=[
                    [1.01, "Premium", "G", "VS2", 62.5, 57.0, 6.35, 6.39, 3.98],
                    [2.5, "Ideal", "D", "VVS1", 60.0, 58.0, 8.72, 8.68, 5.22],
                    [0.5, "Very Good", "I", "SI1", 61.5, 55.0, 3.95, 3.98, 2.43]
                ],
                inputs=[carat, cut, color, clarity, depth, table, x, y, z],
                label="Sample Configurations"
            )

        # Output Section (Right)
        with gr.Column(scale=1, elem_id="prediction-box"):
            gr.Markdown("### 🏷️ Estimated Market Value")
            output_price = gr.Textbox(label="", elem_id="price-text", show_label=False, interactive=False,
                                      value="$0.00")
            output_tier = gr.Textbox(label="Market Tier", interactive=False, value="Pending...")

    predict_btn.click(
        fn=predict,
        inputs=[carat, cut, color, clarity, depth, table, x, y, z],
        outputs=[output_price, output_tier]
    )

    clear_btn.click(
        fn=lambda: [0.5, "Ideal", "E", "VS1", 61.5, 55.0, 5.1, 5.12, 3.15, "$0.00", "Pending..."],
        outputs=[carat, cut, color, clarity, depth, table, x, y, z, output_price, output_tier]
    )

if __name__ == "__main__":
    interface.launch()