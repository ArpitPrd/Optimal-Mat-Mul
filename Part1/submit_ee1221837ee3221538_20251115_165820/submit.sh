#!/usr/bin/env bash
# Package submission: creates a tarball with code + metadata
set -euo pipefail
TEAM=${1:-team_unknown}
OUTPUT=submit_${TEAM}_$(date +%Y%m%d_%H%M%S).tar.gz

#remove previous tgz
rm -f *${TEAM}*gz
# required files
tar -czf ${OUTPUT} README.md baseline optimized Makefile run.sh run_all_tests.sh submit.sh report_template.md plots plot_results.py scorer.py run_tests.sh


echo "Created submission package: ${OUTPUT}"


echo "Please upload ${OUTPUT} in Moodle "
