#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export DS_VIDEOS="${DS_VIDEOS:-/Users/tarunyadgirkar/TarunsCode/ds-hack}"
export WC_DATA="${WC_DATA:-/Users/tarunyadgirkar/TarunsCode/ds-hack}"
PY=.venv/bin/python

$PY variance/compute.py \
  --tasks belly-band-assembly axle-shaft-cutting garment-iron-press \
  --out variance/results/smoke_variance.json \
  --ranking-out variance/results/smoke_ranking.txt

$PY - <<'EOF'
import json

d = json.load(open("variance/results/smoke_variance.json"))
required = {
    "n_clips", "n_families", "n_cameras", "detection_rate", "silhouette_k2", "perm_p",
    "labels", "families", "cameras", "family_confounded", "camera_confounded",
    "excluded", "exclude_reason",
}
assert len(d) == 3, f"expected 3 tasks, got {len(d)}"
for tid, r in d.items():
    missing = required - set(r.keys())
    assert not missing, f"{tid} missing keys {missing}"
    assert isinstance(r["n_clips"], int) and r["n_clips"] > 0
    assert isinstance(r["excluded"], bool)
    assert isinstance(r["family_confounded"], bool)
    assert isinstance(r["camera_confounded"], bool)
    assert isinstance(r["labels"], list)
    assert isinstance(r["families"], list) and isinstance(r["cameras"], list)
    assert len(r["labels"]) == len(r["families"]) == len(r["cameras"])
    if not r["excluded"]:
        assert r["silhouette_k2"] is not None and r["perm_p"] is not None
        assert 0.0 <= r["perm_p"] <= 1.0
        assert -1.0 <= r["silhouette_k2"] <= 1.0
print("schema OK for", sorted(d.keys()))
EOF

echo "SMOKE TEST PASSED"
