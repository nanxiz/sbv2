import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from common.log import logger


def download_bert_models():
    with open("bert/bert_models.json", "r") as fp:
        models = json.load(fp)
    for k, v in models.items():
        local_path = Path("bert").joinpath(k)
        for file in v["files"]:
            if not Path(local_path).joinpath(file).exists():
                logger.info(f"Downloading {k} {file}")
                hf_hub_download(
                    v["repo_id"],
                    file,
                    local_dir=local_path,
                    local_dir_use_symlinks=False,
                )


def download_slm_model():
    local_path = Path("slm/wavlm-base-plus/")
    file = "pytorch_model.bin"
    if not Path(local_path).joinpath(file).exists():
        logger.info(f"Downloading wavlm-base-plus {file}")
        hf_hub_download(
            "microsoft/wavlm-base-plus",
            file,
            local_dir=local_path,
            local_dir_use_symlinks=False,
        )


def download_pretrained_models():
    files = ["G_0.safetensors", "D_0.safetensors", "DUR_0.safetensors"]
    local_path = Path("pretrained")
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            logger.info(f"Downloading pretrained {file}")
            hf_hub_download(
                "litagin/Style-Bert-VITS2-1.0-base",
                file,
                local_dir=local_path,
                local_dir_use_symlinks=False,
            )


def download_jp_extra_pretrained_models():
    files = ["G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"]
    local_path = Path("pretrained_jp_extra")
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            logger.info(f"Downloading JP-Extra pretrained {file}")
            hf_hub_download(
                "litagin/Style-Bert-VITS2-2.0-base-JP-Extra",
                file,
                local_dir=local_path,
                local_dir_use_symlinks=False,
            )

def download_njnj_model():
    local_path = Path("model_assets/njnj/")
    files = ["njnj_e96_s13500_thebest.safetensors", "config.json", "style_vectors.npy"]
    for file in files:

        if not Path(local_path).joinpath(file).exists():
            logger.info(f"Downloading njnj model {file}")
            hf_hub_download(
                "nanxiz/zacabnzh_xxx_tts_0",
                file,
                local_dir=local_path,
                local_dir_use_symlinks=False,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_jvnv", action="store_true")
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="Dataset root path (default: Data)",
        default=None,
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        help="Assets root path (default: model_assets)",
        default=None,
    )
    args = parser.parse_args()

    download_bert_models()

    download_slm_model()

    download_pretrained_models()

    download_jp_extra_pretrained_models()


    if args.dataset_root is None and args.assets_root is None:
        return

    # Change default paths if necessary
    paths_yml = Path("configs/paths.yml")
    with open(paths_yml, "r", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    if args.assets_root is not None:
        yml_data["assets_root"] = args.assets_root
    if args.dataset_root is not None:
        yml_data["dataset_root"] = args.dataset_root
    with open(paths_yml, "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)


if __name__ == "__main__":
    main()
