#!/usr/bin/env bash
# Symlink existing PolyHoop simulation caches into ./data/<subdir>/.
#
# Usage:
#   bash scripts/link_existing_data.sh <data-root>
#
# For every cache subdirectory named in configs/experiment/*.yaml, this
# script creates a symlink ./data/<subdir> -> <data-root>/<subdir> if
# <data-root>/<subdir> exists. Skips subdirs that don't exist at the source.
# Existing symlinks at the destination are replaced; existing real
# directories are left alone (with a warning).
#
# Exit codes:
#   0  success (some subdirs may have been skipped — see the summary)
#   1  invalid arguments or missing data-root

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <data-root>" >&2
  echo "  e.g. $0 /path/to/your/polyhoop/cache_root" >&2
  exit 1
fi

src="$1"
if [[ ! -d "$src" ]]; then
  echo "error: <data-root> does not exist: $src" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
configs_dir="$repo_root/configs/experiment"
data_dir="$repo_root/data"
mkdir -p "$data_dir"

# Extract every distinct cache subdirectory referenced by experiment configs.
# Match `cache_dir: ${oc.env:DATA_DIR,./data}/<subdir>/` entries.
mapfile -t subdirs < <(
  grep -rh 'cache_dir:.*\${oc.env:DATA_DIR' "$configs_dir" \
    | sed -E 's|.*\$\{oc\.env:DATA_DIR,\./data\}/([^/]+)/.*|\1|' \
    | sort -u
)

if [[ ${#subdirs[@]} -eq 0 ]]; then
  echo "warn: no cache_dir entries found under $configs_dir" >&2
  exit 0
fi

linked=0
skipped_missing=0
skipped_exists=0
for sub in "${subdirs[@]}"; do
  src_path="$src/$sub"
  dst_path="$data_dir/$sub"
  if [[ ! -e "$src_path" ]]; then
    skipped_missing=$((skipped_missing + 1))
    continue
  fi
  if [[ -L "$dst_path" ]]; then
    rm "$dst_path"
  elif [[ -e "$dst_path" ]]; then
    echo "skip: $dst_path exists and is not a symlink (leaving it alone)" >&2
    skipped_exists=$((skipped_exists + 1))
    continue
  fi
  ln -s "$src_path" "$dst_path"
  linked=$((linked + 1))
  echo "link: $dst_path -> $src_path"
done

echo
echo "linked $linked subdirectories; missing at source: $skipped_missing; pre-existing real dirs: $skipped_exists"
