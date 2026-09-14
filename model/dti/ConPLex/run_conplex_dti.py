import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.getcwd(), "models", "huggingface"))
import sys


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from conplex_dti.__main__ import main

    main()
