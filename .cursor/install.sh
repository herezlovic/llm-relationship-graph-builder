#!/usr/bin/env bash
# Idempotent environment bootstrap for the LLM Relationship Graph Builder.
# Prepares system packages, a local Neo4j server, the Python backend venv,
# the frontend node_modules, and local .env files. Safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${HOME}/tools"
NEO4J_HOME="${TOOLS_DIR}/neo4j"
NEO4J_VERSION="5.26.0"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"

echo "==> [1/5] System packages"
if ! command -v tesseract >/dev/null 2>&1 || ! command -v java >/dev/null 2>&1 || ! dpkg -s libmagic1 >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libmagic1 libgl1 libglib2.0-0 libglx-mesa0 libgomp1 libreoffice \
    cmake poppler-utils tesseract-ocr git curl \
    openjdk-21-jre-headless python3.12-venv python3.12-dev build-essential
else
  echo "System packages already present; skipping apt install."
fi

echo "==> [2/5] Local Neo4j ${NEO4J_VERSION}"
mkdir -p "${TOOLS_DIR}"
if [ ! -x "${NEO4J_HOME}/bin/neo4j" ]; then
  curl -fsSL -o "${TOOLS_DIR}/neo4j.tar.gz" \
    "https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
  tar -xzf "${TOOLS_DIR}/neo4j.tar.gz" -C "${TOOLS_DIR}"
  rm -rf "${NEO4J_HOME}"
  mv "${TOOLS_DIR}/neo4j-community-${NEO4J_VERSION}" "${NEO4J_HOME}"
  rm -f "${TOOLS_DIR}/neo4j.tar.gz"

  curl -fsSL -o "${NEO4J_HOME}/plugins/apoc.jar" \
    "https://github.com/neo4j/apoc/releases/download/${NEO4J_VERSION}/apoc-${NEO4J_VERSION}-core.jar"

  cat >> "${NEO4J_HOME}/conf/neo4j.conf" <<'EOF'
server.default_listen_address=0.0.0.0
dbms.security.procedures.unrestricted=apoc.*,gds.*
dbms.security.procedures.allowlist=apoc.*,gds.*
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1G
EOF

  "${NEO4J_HOME}/bin/neo4j-admin" dbms set-initial-password "${NEO4J_PASSWORD}"
else
  echo "Neo4j already installed at ${NEO4J_HOME}; skipping download."
fi

echo "==> [3/5] Backend Python dependencies"
cd "${REPO_ROOT}/backend"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt

# Pre-download the default sentence-transformer embedding model into ./local_model
# (mirrors backend/Dockerfile) so /connect and extraction work offline-first.
if [ ! -d "local_model" ]; then
  python -c "from transformers import AutoTokenizer, AutoModel; \
name='sentence-transformers/all-MiniLM-L6-v2'; \
AutoTokenizer.from_pretrained(name).save_pretrained('./local_model'); \
AutoModel.from_pretrained(name).save_pretrained('./local_model')"
fi
python -m nltk.downloader -d "${HOME}/nltk_data" punkt punkt_tab averaged_perceptron_tagger >/dev/null 2>&1 || true
deactivate

echo "==> [4/5] Frontend node dependencies"
cd "${REPO_ROOT}/frontend"
yarn install --frozen-lockfile

echo "==> [5/5] Local .env files"
if [ ! -f "${REPO_ROOT}/backend/.env" ]; then
  cat > "${REPO_ROOT}/backend/.env" <<EOF
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="${NEO4J_PASSWORD}"
NEO4J_DATABASE="neo4j"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
EMBEDDING_PROVIDER="sentence-transformer"
IS_EMBEDDING="TRUE"
KNN_MIN_SCORE="0.94"
UPDATE_GRAPH_CHUNKS_PROCESSED="20"
ENTITY_EMBEDDING="False"
GCS_FILE_CACHE="False"
GCP_LOG_METRICS_ENABLED="False"
NUMBER_OF_CHUNKS_TO_COMBINE="6"
MAX_TOKEN_CHUNK_SIZE="10000"
DUPLICATE_SCORE_VALUE="0.97"
DUPLICATE_TEXT_DISTANCE="3"

# Add an LLM key here to exercise extraction/chat, e.g.:
# OPENAI_API_KEY="sk-..."
# LLM_MODEL_CONFIG_openai_gpt_5_mini="gpt-5-mini,sk-..."
EOF
  echo "Wrote backend/.env"
else
  echo "backend/.env exists; leaving as-is."
fi

if [ ! -f "${REPO_ROOT}/frontend/.env" ]; then
  cp "${REPO_ROOT}/frontend/example.env" "${REPO_ROOT}/frontend/.env"
  echo "Wrote frontend/.env from example.env"
else
  echo "frontend/.env exists; leaving as-is."
fi

echo "==> Install complete."
