#!/usr/bin/env bash
# Per-boot startup: ensure the local Neo4j server is running and ready.
# Backend and frontend dev servers run as visible terminals (see environment.json).
set -euo pipefail

NEO4J_HOME="${HOME}/tools/neo4j"

if [ ! -x "${NEO4J_HOME}/bin/neo4j" ]; then
  echo "Neo4j not installed at ${NEO4J_HOME}; run .cursor/install.sh first." >&2
  exit 1
fi

export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")}"

# neo4j start is a no-op if already running; it returns after launching the daemon.
"${NEO4J_HOME}/bin/neo4j" start

echo "Waiting for Neo4j Bolt on localhost:7687 ..."
for _ in $(seq 1 60); do
  if "${NEO4J_HOME}/bin/cypher-shell" -a bolt://localhost:7687 \
       -u neo4j -p "${NEO4J_PASSWORD:-password123}" "RETURN 1;" >/dev/null 2>&1; then
    echo "Neo4j is ready."
    exit 0
  fi
  sleep 2
done

echo "Neo4j did not become ready in time." >&2
exit 1
