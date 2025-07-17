python -m venv gemma-env
source gemma-env/bin/activate
pip install packaging
pip install wheel
pip install torch
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

