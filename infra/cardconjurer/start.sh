#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="${CARDCONJURER_SRC:-/tmp/cardconjurer-master}"
if [ ! -d "$SRC_DIR" ]; then
  echo "missing $SRC_DIR — see README" >&2
  exit 1
fi
docker rm -f cardconjurer 2>/dev/null || true
docker run -d --rm --name cardconjurer -p 4242:4242 \
  -v "$SRC_DIR:/usr/share/nginx/html:ro" \
  -v "$SRC_DIR/app.conf:/etc/nginx/nginx.conf:ro" \
  nginx:alpine
echo "waiting for cardconjurer ..."
until curl -sf -o /dev/null http://localhost:4242/; do sleep 1; done
echo "ready: http://localhost:4242/creator/"
