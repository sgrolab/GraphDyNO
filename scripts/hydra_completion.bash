#!/usr/bin/env bash
# Fast Hydra tab completion for decoder_gnn project.
# Instead of invoking Python (which imports torch, etc.), this script
# walks the configs/ directory tree and completes config groups, config
# names, experiment names, and dotpath overrides from cached YAML keys.
#
# Usage:
#   source scripts/hydra_completion.bash
#
# This registers completions for: train.py, dino/train_dino.py,
# dino/precompute_embeddings.py, dino/cell_mask_extractor.py,
# and the "python" command when followed by one of those scripts.

_DECODER_GNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Build a flat list of dotpath keys from a YAML file (no dependencies).
# Handles nested keys, skips comments/blanks/list items/defaults blocks.
# ---------------------------------------------------------------------------
_hydra_yaml_keys() {
    local file="$1"
    local prefix="$2"
    awk -v prefix="$prefix" '
    BEGIN { depth = 0; stack[0] = ""; in_defaults = 0 }
    /^[[:space:]]*#/  { next }
    /^[[:space:]]*$/  { next }
    {
        # Count leading spaces (assume 2-space indent)
        match($0, /^[[:space:]]*/);
        indent = RLENGTH;
        level = int(indent / 2);

        # Strip leading whitespace
        sub(/^[[:space:]]+/, "");

        # Skip defaults block entries
        if ($0 ~ /^defaults:/) { in_defaults = 1; defaults_level = level; next }
        if (in_defaults) {
            if (level > defaults_level) { next }
            else { in_defaults = 0 }
        }

        # Skip list items (- foo) and Hydra resolver lines (${...})
        if ($0 ~ /^-/) next;

        # Match key: value or key:
        if (match($0, /^([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*:/, arr)) {
            key = substr($0, RSTART, RLENGTH - 1);
            sub(/:$/, "", key);
            sub(/[[:space:]]+$/, "", key);
            stack[level] = key;

            # Build full dotpath
            path = "";
            for (i = 0; i <= level; i++) {
                if (i > 0) path = path ".";
                path = path stack[i];
            }
            if (prefix != "") path = prefix "." path;
            print path "=";
        }
    }
    ' "$file"
}

# ---------------------------------------------------------------------------
# Cache: build dotpath keys from all YAML files once per shell session,
# refresh if any YAML file is newer than the cache.
# ---------------------------------------------------------------------------
_HYDRA_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hydra_completion"
_HYDRA_KEYS_CACHE="$_HYDRA_CACHE_DIR/keys.cache"
_HYDRA_GROUPS_CACHE="$_HYDRA_CACHE_DIR/groups.cache"

_hydra_rebuild_cache() {
    mkdir -p "$_HYDRA_CACHE_DIR"
    local configs_dir="$_DECODER_GNN_ROOT/configs"

    # --- Config groups cache ---
    # Format: group=name (e.g., data=polyhoop, experiment=oscillating_cell)
    {
        # Top-level groups
        for group_dir in "$configs_dir"/*/; do
            [ -d "$group_dir" ] || continue
            group="$(basename "$group_dir")"
            # Skip hydra internal dirs
            [[ "$group" == "__pycache__" ]] && continue
            for f in "$group_dir"/*.yaml; do
                [ -f "$f" ] || continue
                name="$(basename "$f" .yaml)"
                echo "${group}=${name}"
            done
        done
        # Nested groups (e.g., dino/model)
        for group_dir in "$configs_dir"/*/*/; do
            [ -d "$group_dir" ] || continue
            parent="$(basename "$(dirname "$group_dir")")"
            group="$(basename "$group_dir")"
            for f in "$group_dir"/*.yaml; do
                [ -f "$f" ] || continue
                name="$(basename "$f" .yaml)"
                echo "${parent}/${group}=${name}"
            done
        done
    } > "$_HYDRA_GROUPS_CACHE"

    # --- Dotpath keys cache ---
    # For each config group's default yaml, extract dotpath keys
    # with the group name as prefix (matching Hydra's merged config structure).
    {
        # Main config groups -> keys prefixed with group name
        for group_dir in "$configs_dir"/*/; do
            [ -d "$group_dir" ] || continue
            group="$(basename "$group_dir")"
            [[ "$group" == "__pycache__" || "$group" == "experiment" ]] && continue
            # Use all yaml files in the group for maximum coverage
            for f in "$group_dir"/*.yaml; do
                [ -f "$f" ] || continue
                _hydra_yaml_keys "$f" "$group"
            done
        done

        # Nested groups (dino/*) -> keys prefixed with parent.group
        for group_dir in "$configs_dir"/*/*/; do
            [ -d "$group_dir" ] || continue
            parent="$(basename "$(dirname "$group_dir")")"
            group="$(basename "$group_dir")"
            for f in "$group_dir"/*.yaml; do
                [ -f "$f" ] || continue
                _hydra_yaml_keys "$f" "${parent}.${group}"
            done
        done

        # Experiment configs use @package _global_ so keys are top-level
        for f in "$configs_dir"/experiment/*.yaml; do
            [ -f "$f" ] || continue
            _hydra_yaml_keys "$f" ""
        done
    } | sort -u > "$_HYDRA_KEYS_CACHE"
}

_hydra_ensure_cache() {
    local configs_dir="$_DECODER_GNN_ROOT/configs"
    # Rebuild if cache is missing or any yaml is newer
    if [ ! -f "$_HYDRA_KEYS_CACHE" ] || [ ! -f "$_HYDRA_GROUPS_CACHE" ]; then
        _hydra_rebuild_cache
        return
    fi
    # Check if any yaml file is newer than cache
    local newest
    newest="$(find "$configs_dir" -name '*.yaml' -newer "$_HYDRA_KEYS_CACHE" -print -quit 2>/dev/null)"
    if [ -n "$newest" ]; then
        _hydra_rebuild_cache
    fi
}

# ---------------------------------------------------------------------------
# Hydra flags that take no argument
# ---------------------------------------------------------------------------
_HYDRA_FLAGS=(
    --help -h
    --hydra-help
    --version
    --cfg -c
    --package -p
    --info -i
    --multirun -m
)

# ---------------------------------------------------------------------------
# Extract the real current token from COMP_LINE, ignoring COMP_WORDBREAKS
# splitting at '='. Bash normally splits "experiment=foo" into two words at
# the '=', so COMP_WORDS/COMP_CWORD give us the wrong token.
# ---------------------------------------------------------------------------
_hydra_get_cur() {
    local line="${COMP_LINE:0:COMP_POINT}"
    # Extract the last whitespace-delimited token
    echo "${line##* }"
}

# ---------------------------------------------------------------------------
# Main completion function
# ---------------------------------------------------------------------------
_hydra_complete() {
    COMPREPLY=()
    local cur
    cur="$(_hydra_get_cur)"

    _hydra_ensure_cache

    # Complete Hydra flags
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "${_HYDRA_FLAGS[*]}" -- "$cur") )
        return
    fi

    # Strip leading + or ~ prefix for matching, remember it for output
    local prefix_char=""
    local match_cur="$cur"
    if [[ "$cur" == +* ]] || [[ "$cur" == ~* ]]; then
        prefix_char="${cur:0:1}"
        match_cur="${cur:1}"
    fi

    # If there's an = sign, complete the value after it.
    # IMPORTANT: Because COMP_WORDBREAKS contains '=', readline treats
    # everything after '=' as the current word. So we must return only the
    # value part (after '='), not the full "key=value" string — readline
    # will prepend "key=" itself.
    if [[ "$match_cur" == *=* ]]; then
        local key="${match_cur%%=*}"
        local val_prefix="${match_cur#*=}"

        # Check if this is a config group override (e.g., experiment=osc<TAB>)
        local group_values
        group_values="$(grep "^${key}=" "$_HYDRA_GROUPS_CACHE" 2>/dev/null | sed "s|^${key}=||")"
        if [ -n "$group_values" ]; then
            COMPREPLY=( $(compgen -W "$group_values" -- "$val_prefix") )
        fi
        return
    fi

    # No = sign yet: complete group names and dotpath keys
    local words=""

    # Config group names
    if [ -f "$_HYDRA_GROUPS_CACHE" ]; then
        local groups
        groups="$(cut -d= -f1 "$_HYDRA_GROUPS_CACHE" | sort -u)"
        if [ -n "$prefix_char" ]; then
            groups="$(echo "$groups" | sed "s/^/${prefix_char}/")"
        fi
        words="$words $groups"
    fi

    # Dotpath keys (only when no prefix char — +/~ are for group overrides)
    if [ -z "$prefix_char" ] && [ -f "$_HYDRA_KEYS_CACHE" ]; then
        words="$words $(cat "$_HYDRA_KEYS_CACHE")"
    fi

    COMPREPLY=( $(compgen -W "$words" -- "$cur") )

    # If unique match is a config group name (no =), append =
    if [ ${#COMPREPLY[@]} -eq 1 ]; then
        local match="${COMPREPLY[0]}"
        if [[ "$match" != *=* ]]; then
            # Check if it's a known group (strip prefix char for lookup)
            local lookup="${match#[+~]}"
            if grep -q "^${lookup}=" "$_HYDRA_GROUPS_CACHE" 2>/dev/null; then
                COMPREPLY=( "${match}=" )
                compopt -o nospace
                return
            fi
        fi
    fi

    # Suppress trailing space for any match containing =
    local has_eq=false
    for r in "${COMPREPLY[@]}"; do
        [[ "$r" == *= ]] && has_eq=true && break
    done
    if $has_eq; then
        compopt -o nospace
    fi
}

# ---------------------------------------------------------------------------
# Register completions
# ---------------------------------------------------------------------------
_hydra_complete_python() {
    # Only activate when the python command is followed by a known Hydra script
    local script=""
    for ((i=1; i<COMP_CWORD; i++)); do
        case "${COMP_WORDS[i]}" in
            train.py|./train.py|train_autoencoder.py|./train_autoencoder.py|\
            scripts/generate_cache.py|dino/train_dino.py|./dino/train_dino.py|\
            precompute_embeddings.py|./precompute_embeddings.py|\
            dino/precompute_embeddings.py|./dino/precompute_embeddings.py|\
            dino/cell_mask_extractor.py|./dino/cell_mask_extractor.py)
                script="${COMP_WORDS[i]}"
                break
                ;;
        esac
    done
    if [ -n "$script" ]; then
        _hydra_complete
    else
        # Fall back to default python completion
        return 124
    fi
}

# Register for direct script invocation
complete -F _hydra_complete train.py
complete -F _hydra_complete ./train.py
complete -F _hydra_complete scripts/generate_cache.py
complete -F _hydra_complete precompute_embeddings.py
complete -F _hydra_complete ./precompute_embeddings.py
complete -F _hydra_complete train_autoencoder.py
complete -F _hydra_complete ./train_autoencoder.py
complete -F _hydra_complete scripts/generate_cache.py
complete -F _hydra_complete ./scripts/generate_cache.py

# Register for python invocation (-o default falls back to filename completion
# when no Hydra script has been detected yet, e.g. "python tr<TAB>")
complete -o default -F _hydra_complete_python python
complete -o default -F _hydra_complete_python python3

echo "Hydra fast completion loaded for decoder_gnn."
echo "  Completions registered for: train.py, python train.py, etc."
echo "  Config cache: $_HYDRA_CACHE_DIR"
echo "  Run '_hydra_rebuild_cache' to force refresh."
