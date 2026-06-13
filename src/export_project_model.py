import argparse
import os
import sys
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Export this project's Bert-VITS2 checkpoint for the C++ runner."
    )
    parser.add_argument("model", nargs="?", default="Data/models/G_46000.pth")
    parser.add_argument("--config")
    parser.add_argument("--name", default="model")
    parser.add_argument("--novq", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    from onnx_modules import export_onnx

    config = args.config
    if config is None:
        model_dir = os.path.dirname(os.path.abspath(args.model))
        candidates = [
            os.path.join(model_dir, "config.json"),
            os.path.join("Data", "config.json"),
        ]
        config = next((path for path in candidates if os.path.exists(path)), None)
        if config is None:
            raise FileNotFoundError(
                "config.json was not found next to the G.pth or at Data/config.json; "
                "pass --config explicitly."
            )

    os.makedirs(os.path.join("onnx", args.name), exist_ok=True)
    export_onnx(args.name, args.model, config, args.novq, args.dev)
    shutil.copy2(config, os.path.join("onnx", args.name, "config.json"))
    print(f"exported to onnx/{args.name}")
    print(f"used config {config}")


if __name__ == "__main__":
    main()
