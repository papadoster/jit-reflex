#!/usr/bin/env bash
# E1b, pre-registered after E1 = KILL (spec section 5, E1b): correction clipped to +-1 as in E2, fresh seed 1000.
exec env OUT=results/probe_e1b "$(dirname "$0")/run_e1.sh" --seed 1000 --verdict-on lin_clip "$@"
