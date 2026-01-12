pip install uv

uv sync --frozen \
    --extra apo \
    --extra verl \
    --group dev \
    --group torch-cpu \
    --group torch-stable \
    --group trl \
    --group agents \
    --no-default-groups

uv pip install poetry

git clone https://github.com/ultmaster/litellm.git -b fix/anthropic-to-openai-missing-text-content

cd litellm

poetry install --with dev --extras proxy

cd ../

cd examples/cc

uv pip install -r requirements.txt

