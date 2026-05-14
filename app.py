"""
Gradio demo for Turner / Constable art classifier.

Features:
  - Upload a painting → get artist prediction + confidence bars
  - Grad-CAM heatmap overlay showing which regions drove the decision
  - Toggle TTA for improved accuracy

Usage:
    python app.py --checkpoint checkpoints/resnet50/best.pth
    python app.py --checkpoint checkpoints/resnet50/best.pth --share
    python app.py --checkpoint checkpoints/swin_base_patch4_window7_224/best.pth \
        --model-name swin_base_patch4_window7_224 --image-size 384 --share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import gradio as gr
except ImportError:
    sys.exit("Install gradio: pip install gradio")

from src.inference.predict import (
    generate_gradcam,
    load_model,
    predict_single,
    predict_with_tta,
)

ARTISTS = ["Turner", "Constable"]
ARTIST_DESCRIPTIONS = {
    "Turner": (
        "J.M.W. Turner (1775–1851) was known for his atmospheric, almost impressionistic "
        "landscapes and seascapes. His works feature dramatic light, sweeping skies, and "
        "a hazy, luminous quality. He was a pioneer in capturing transient natural phenomena."
    ),
    "Constable": (
        "John Constable (1776–1837) was celebrated for his naturalistic depictions of the "
        "English countryside. His paintings feature detailed foliage, realistic cloud studies, "
        "and a cooler, more grounded palette — particularly the Suffolk countryside."
    ),
}


def build_predict_fn(model, device, image_size: int):
    """Returns a function compatible with Gradio's interface."""

    def predict(image, use_tta: bool, show_gradcam: bool):
        if image is None:
            return "Please upload an image.", None, None

        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image)

        # Prediction
        if use_tta:
            result = predict_with_tta(pil_image, model, device, image_size)
        else:
            result = predict_single(pil_image, model, device, image_size)

        artist = result["artist"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        # Build output text
        label_text = (
            f"**{artist}**\n"
            f"Confidence: {confidence*100:.1f}%\n\n"
            f"Probabilities:\n"
            f"- Turner: {probs['turner']*100:.1f}%\n"
            f"- Constable: {probs['constable']*100:.1f}%\n\n"
            f"---\n{ARTIST_DESCRIPTIONS[artist]}"
        )

        if use_tta:
            label_text += f"\n\n*(Test-Time Augmentation used: {result.get('n_augments', 5)} views)*"

        # Confidence bar values for Gradio Label component
        confidence_bars = {
            "Turner": float(probs["turner"]),
            "Constable": float(probs["constable"]),
        }

        # Grad-CAM heatmap
        cam_overlay = None
        if show_gradcam:
            try:
                target_class = 0 if artist == "Turner" else 1
                _, overlay = generate_gradcam(pil_image, model, device, target_class, image_size)
                cam_overlay = overlay
            except Exception as e:
                print(f"Grad-CAM failed: {e}")

        return label_text, confidence_bars, cam_overlay

    return predict


def build_interface(model, device, image_size: int) -> gr.Blocks:
    predict_fn = build_predict_fn(model, device, image_size)

    with gr.Blocks(title="Turner or Constable?", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Turner or Constable?\n"
            "Upload a painting to find out whether it was made by **J.M.W. Turner** "
            "or **John Constable** — or which artist's style it most resembles.\n\n"
            "*Trained on WikiArt paintings with a 3-phase fine-tuning strategy "
            "and differential learning rates.*"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Painting", type="numpy")
                with gr.Row():
                    tta_toggle = gr.Checkbox(label="Use TTA (more accurate, slower)", value=False)
                    gradcam_toggle = gr.Checkbox(label="Show Grad-CAM heatmap", value=True)
                predict_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                text_output = gr.Markdown(label="Result")
                label_output = gr.Label(label="Confidence", num_top_classes=2)
                cam_output = gr.Image(label="Grad-CAM: regions that influenced the decision", type="numpy")

        predict_btn.click(
            fn=predict_fn,
            inputs=[image_input, tta_toggle, gradcam_toggle],
            outputs=[text_output, label_output, cam_output],
        )

        # Example images (will be populated if examples/ dir exists)
        examples_dir = Path("examples")
        if examples_dir.exists():
            example_paths = list(examples_dir.glob("*.jpg"))[:6]
            if example_paths:
                gr.Examples(
                    examples=[[str(p), False, True] for p in example_paths],
                    inputs=[image_input, tta_toggle, gradcam_toggle],
                )

        gr.Markdown(
            "---\n"
            "**About the artists:**\n\n"
            "- **Turner** was known for atmospheric, luminous landscapes and seascapes "
            "(Venice, the Thames, dramatic skies).\n"
            "- **Constable** painted naturalistic English countryside with detailed foliage "
            "and realistic cloud formations (Dedham Vale, Constable Country).\n\n"
            "*Note: Both artists were contemporaries painting British landscapes — "
            "the classifier must learn subtle stylistic differences in palette, "
            "brushwork texture, and atmospheric treatment.*"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Turner/Constable Gradio demo")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth checkpoint")
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, device = load_model(args.checkpoint, args.model_name)
    print(f"Model ready on {device}")

    demo = build_interface(model, device, args.image_size)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
