#!/usr/bin/env bash
# kind end-to-end test: run the 2-node training Job, delete a trainer pod in the
# middle of training, and verify the job resumes from a checkpoint and completes.
set -euo pipefail
EVENTS=${EVENTS:-/tmp/rtrain-ckpt/run/events.jsonl}

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
pass() { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m  ✗ %s\033[0m\n' "$*"; kubectl get pods -o wide || true; kubectl logs -l app=trainer --tail=30 --prefix || true; exit 1; }

last_step() { [[ -f "$EVENTS" ]] && python3 -c "
import json,sys
s=[json.loads(l).get('step',0) for l in open('$EVENTS') if '\"train\"' in l]
print(max(s) if s else 0)" || echo 0; }

log "Starting the training Job"
kubectl apply -f deploy/k8s/trainer-job.yaml
start=$SECONDS
until (( $(last_step) >= 150 )); do
  (( SECONDS - start > 300 )) && fail "training did not reach step 150"
  sleep 2
done
pass "training reached step $(last_step)"

victim=$(kubectl get pods -l job-name=trainer,batch.kubernetes.io/job-completion-index=1 -o jsonpath='{.items[0].metadata.name}')
log "Deleting trainer pod $victim at step $(last_step)"
kubectl delete pod "$victim" --wait=false

log "Waiting for the Job to complete"
kubectl wait --for=condition=complete job/trainer --timeout=600s || fail "job did not complete"

python3 - "$EVENTS" <<'EOF' || fail "recovery checks failed"
import json, sys
ev = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
starts = [e for e in ev if e["event"] == "attempt_start"]
done = [e for e in ev if e["event"] == "completed"]
print(f"  attempts: {[s['resumed_from'] for s in starts]}")
assert len(starts) >= 2, "expected at least one restart"
assert any(s["resumed_from"] > 0 for s in starts[1:]), "a restart must resume from a checkpoint, not step 0"
assert done and done[-1]["step"] == 600, "training must finish all steps"
print(f"  completed step 600, eval_loss={done[-1]['eval_loss']}")
EOF
pass "pod deleted mid-training; job resumed from a checkpoint and completed"
echo -e "\nE2E PASSED"
