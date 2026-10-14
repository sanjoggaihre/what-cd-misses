#!/bin/bash
# preflight_ablation.sh -- run every check before starting run_ablation_2gpu.sh.
# Run from the Point-MAE folder on the H100 node:   bash preflight_ablation.sh
# Paste the whole output back if anything says FAIL.

cd "$(dirname "$0")" 2>/dev/null || true
fail=0
say() { printf '%-6s %s\n' "$1" "$2"; }

echo "== 1. runner script"
if bash -n run_ablation_2gpu.sh; then say ok "run_ablation_2gpu.sh syntax"; else say FAIL "syntax error in run_ablation_2gpu.sh"; fail=1; fi
left=$(grep -v '^#' run_ablation_2gpu.sh | grep -n "REPLACE_" || true)
if [ -z "$left" ]; then say ok "no REPLACE_ placeholders left in the commands"
else say FAIL "placeholders still to fill:"; echo "$left"; fail=1; fi
grep -E '^(W005_CFG|W000_CFG|DCD_CFG|REP_CFG|OUT|OUT_GEN)=' run_ablation_2gpu.sh

echo; echo "== 2. config files"
for c in cfgs/pretrain_pepper_baseline_dloss.yaml cfgs/dloss_w000.yaml cfgs/abl_cov_only.yaml \
         cfgs/abl_sp_only.yaml cfgs/abl_cov_masked.yaml cfgs/dcd_a50.yaml cfgs/rep_h007_w1.yaml; do
  if [ -f "$c" ]; then say ok "$c"; else say FAIL "$c missing"; fail=1; fi
done

echo; echo "== 3. code change in models/Point_MAE.py"
if python -c "import ast; ast.parse(open('models/Point_MAE.py').read())"; then say ok "Point_MAE.py parses"
else say FAIL "Point_MAE.py has a syntax error"; fail=1; fi
grep -n -E "cov_target|hole_loss\(|rb_w =" models/Point_MAE.py

echo; echo "== 4. GPUs, disk, existing experiment names"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}'"
ngpu=$(nvidia-smi -L | wc -l)
if [ "$ngpu" -ge 2 ]; then say ok "$ngpu GPUs visible"; else say FAIL "only $ngpu GPU(s) visible here"; fail=1; fi
df -h . | tail -1
for e in experiments/pretrain_pepper_baseline_dloss/cfgs/w005_h100 experiments/dloss_w000/cfgs/w000_h100 \
         experiments/abl_cov_only experiments/abl_sp_only experiments/abl_cov_masked \
         experiments/dcd_a50/cfgs/dcd_a50_l1_seed1 experiments/rep_h007_w1/cfgs/rep_h007_w1_seed1; do
  if [ -e "$e" ]; then say WARN "$e already exists (a new run would mix with it)"; fi
done
[ -d logs ] || mkdir -p logs

echo; echo "== 5. one forward/backward pass per config (about a minute)"
if python smoke_test_ablation.py; then say ok "smoke test"; else say FAIL "smoke test"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASSED -- safe to start run_ablation_2gpu.sh"
else echo "SOME CHECKS FAILED -- fix these before starting"; fi