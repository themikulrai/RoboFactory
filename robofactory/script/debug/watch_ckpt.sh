#!/bin/bash
# Watch 300.ckpt for changes and snapshot each one with the timestamp.
# All 5 retrains save to the same path; this prevents overwrite loss.
set -u
CKPT=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt
DST=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup
mkdir -p "$DST"
LAST=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
echo "[$(date +%H:%M:%S)] watching $CKPT (last mtime=$LAST)"

while true; do
    sleep 20
    NEW=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
    if [ "$NEW" != "$LAST" ] && [ "$NEW" != "0" ]; then
        TS=$(date -d @$NEW +"%H%M%S")
        OUT="$DST/300_at_${TS}.ckpt"
        cp -p "$CKPT" "$OUT"
        echo "[$(date +%H:%M:%S)] snapshot saved: $(basename $OUT) ($(du -h $OUT | cut -f1))"
        LAST=$NEW
    fi
done
