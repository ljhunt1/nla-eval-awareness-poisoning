#!/bin/bash
# Trial 7: the seed-1 x2 arms start ~6.5 h in (after the long continuations), too late to reach step 900 inside the
# 12 h watchdog. Stop each once its step-600 checkpoint is saved; rl_arm then batteries steps 300 and 600.
R=/root/runs7t
for n in Ox2_nf_s1 Ox2d_nf_s1; do
  ( while true; do
      c=$R/$n/critic_step600.pt
      if [ -e $c ] && [ -z "$(find $c -mmin -1)" ]; then
        for p in $(pgrep -f "out $R/$n "); do kill $p; done
        echo "$(date -u +%FT%T) stopped $n at step 600" >> /root/logs/stage.log; break
      fi
      sleep 60
    done ) &
done
wait
