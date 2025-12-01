#!/bin/bash

begin=$(date --iso-8601=seconds)
sleep 70
end=$(date --iso-8601=seconds)

time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))

printf '%02d:%02d:%02d\n' \
  $(( time_taken / 3600 )) \
  $(( (time_taken % 3600) / 60 )) \
  $(( time_taken % 60 ))

