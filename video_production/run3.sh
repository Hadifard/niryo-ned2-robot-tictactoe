#!/bin/sh
cd /home/claude/film
i=0
for a in 0 540 1080 1620 2160 2700 3240 3780 4320 4860 5400 5940; do
  if [ ! -f chunk3_$i.ok ]; then
    b=$((a+540)); python3 scenes3.py chunk $a $b chunk3_$i.mp4 > chunk3_$i.log 2>&1 && touch chunk3_$i.ok
  fi
  i=$((i+1))
done
echo done > alldone3.txt
