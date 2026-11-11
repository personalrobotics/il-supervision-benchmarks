#!/bin/bash
# Monitoring only: held-out InfoNCE (and the sampled action's L2) and a 50-episode coverage of each saved checkpoint
# of the IBC gate, written here. Never read by the gate; the gate's selection and evaluation are its own.
W=$1; G=$2; P=$3; PY=$HOME/code/CCIL-pusht/.lerobot-ibc/bin/python
cd $P
while true; do
  for s in 1 2 3; do
    for d in $G/units/bc-s$s/train/checkpoints/[0-9]*; do
      [ -f $d/pretrained_model/model.safetensors ] || continue
      [ $(( $(date +%s) - $(stat -c %Y $d/pretrained_model/model.safetensors) )) -gt 30 ] || continue
      k=$(basename $d); id=bc-s$s-$k
      [ -f $id.done ] && continue
      echo "{\"$k\": \"$d/pretrained_model\"}" > $id-ckpt.json
      CUDA_VISIBLE_DEVICES=1 $PY -W ignore $W/tools/pusht_v2/select_infonce.py $G/derived/full $G/derived/held_out.json $id-ckpt.json $id-scores.json > $id-select.log 2>&1
      CUDA_VISIBLE_DEVICES=1 $HOME/code/CCIL-pusht/.lerobot-ibc/bin/lerobot-eval --policy.path=$d/pretrained_model --env.type=pusht_history --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=50 --eval.use_async_envs=false --seed=1000 --policy.device=cuda --output_dir=$P/$id-eval > $id-eval.log 2>&1
      $PY -c "
import json
s=json.load(open('$id-scores.json')); c=s['infonce']['$k']; l=s['mean_l2_diagnostic']['$k']
m=json.load(open('$P/$id-eval/eval_info.json'))['per_task'][0]['metrics']
print(f'PEEK $id: held-out InfoNCE {c:.3f}, sampled-action L2 {l:.2f} px, coverage {sum(m[\"max_rewards\"])/len(m[\"max_rewards\"]):.4f} (50 eps), success {100*sum(m[\"successes\"])/len(m[\"successes\"]):.0f}%')
" >> peek.log 2>&1 || echo "PEEK $id: FAILED, see $P/$id-*.log" >> peek.log
      touch $id.done
    done
  done
  if grep -q ^EXIT= $G.log; then
    left=0; for d in $G/units/bc-s*/train/checkpoints/[0-9]*; do [ -f $d/pretrained_model/model.safetensors ] && [ ! -f bc-s$(basename $(dirname $(dirname $(dirname $d))) | sed s/bc-s//)-$(basename $d).done ] && left=1; done
    [ $left = 0 ] && { echo "PEEK finished" >> peek.log; exit 0; }
  fi
  sleep 30
done
