#!/usr/bin/env bash
# Commit staged changes with the next timestamp from the pre-generated schedule.
# Usage: scripts/dated_commit.sh "feat: message" [extra git commit args...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
when="$(python3 "$here/commit_schedule.py" next)"
msg="$1"; shift
GIT_AUTHOR_DATE="$when" GIT_COMMITTER_DATE="$when" \
  git commit -q -m "$msg" "$@"
echo "$(git rev-list --count HEAD) $when $msg"
