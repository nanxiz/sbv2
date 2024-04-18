import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

import torch
from scipy.io import wavfile

import runpod

from common.constants import (
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from common.log import logger
from common.tts_model import Model, ModelHolder
from config import config

ln = "EN"


def load_models(model_holder: ModelHolder):
    model_holder.models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = Model(
            model_path=model_paths[0],
            config_path=model_holder.root_dir / model_name / "config.json",
            style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
            device=model_holder.device,
        )
        model.load_net_g()
        model_holder.models.append(model)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
    )
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path(args.dir)
    model_holder = ModelHolder(model_dir, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info("Loading models...")
    load_models(model_holder)

    return model_holder


model_holder = setup()


def handler(event) -> BytesIO:
    input_data = event.get("input", {})
    text = input_data.get("text", "")
    if not text:
        raise ValueError("Text input is required.")

    model = model_holder.models[0] 

    sr, audio = model.infer(
        text=text,
        language=ln,
        sid=0,  
        reference_audio_path=None,
        sdp_ratio=DEFAULT_SDP_RATIO,
        noise=DEFAULT_NOISE,
        noisew=DEFAULT_NOISEW,
        length=DEFAULT_LENGTH,
        line_split=False,
        split_interval=0,
        assist_text=None,
        assist_text_weight=0,
        use_assist_text=False,
        style="Neutral",
        style_weight=0,
    )
    logger.success("Audio data generated successfully")

    wav_content = BytesIO()
    wavfile.write(wav_content, sr, audio)
    wav_content.seek(0)
    return wav_content

runpod.serverless.start({"handler": handler})