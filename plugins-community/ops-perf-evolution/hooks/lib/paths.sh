#!/usr/bin/env bash
# paths.sh — path safety validators shared by Bash hook and Stop hook.
#
# Mirrors the "三档校验" rules from .claude/agents/lingxi-evo.md L35-47.
# Detects three classes of accidents commonly hit by LLM-generated bash:
#   B1: command uses ${VAR}/* or ${VAR}/... as a path argument and VAR is empty
#   B2: ${VAR} resolves to "/" or "$HOME"
#   (B3 — writing outside EVO_DIR — is a Stop-hook responsibility, not here)

if [[ -n "${LOOP_HOOK_PATHS_SOURCED:-}" ]]; then
    return 0
fi
LOOP_HOOK_PATHS_SOURCED=1

# Risky command patterns. Match the leading token after the command.
# Matches: cp [-rR]*, mv, rm [-rR]*f*
_PATH_SAFETY_RISKY_COMMANDS_RE='^(cp|mv|rm)\b'

# Tokens that signal a dangerous root path
_PATH_SAFETY_ROOT_TOKENS=("/" "/*" "//" "/home" "/root")

# Given a single bash command string, return:
#   0 if safe (or not a risky command)
#   1 if a B1/B2 violation is found; echoes a one-line reason to stderr.
#
# This is intentionally conservative — only blocks when a variable name is
# explicit (${VAR} or $VAR) AND the variable is unset/empty in the current env,
# or when a literal "/" / "/*" is used as the first path arg of rm -rf.
check_bash_path_safety() {
    local cmd="$1"
    [[ -z "$cmd" ]] && return 0

    # Quick exit if the command doesn't start with a risky command
    local first_word
    first_word="$(echo "$cmd" | awk '{print $1}')"
    if ! [[ "${first_word}" =~ ${_PATH_SAFETY_RISKY_COMMANDS_RE} ]]; then
        return 0
    fi

    # Extract all $VAR / ${VAR} references in the command
    # Use grep -oE to pull them out, then check each
    local vars
    vars="$(echo "$cmd" | grep -oE '\$\{?[A-Za-z_][A-Za-z0-9_]*\}?' | sort -u)"

    local var_name
    while IFS= read -r raw; do
        [[ -z "${raw}" ]] && continue
        # strip $ and {} to get the bare name
        var_name="${raw#\$}"
        var_name="${var_name#\{}"
        var_name="${var_name%\}}"

        # Use indirect expansion to read the var from env
        local var_value
        var_value="${!var_name:-}"

        # B1: VAR is empty AND command uses VAR-relative path expansion
        # Pattern: $VAR/* OR ${VAR}/* OR $VAR/<word> OR ${VAR}/<word>
        if [[ -z "${var_value}" ]]; then
            if echo "$cmd" | grep -qE "\\\$\\{?${var_name}\\}?(/|$)"; then
                echo "B1: variable \$${var_name} is empty but used as a path in '${first_word}'" >&2
                return 1
            fi
        fi

        # B2: VAR resolves to a dangerous root
        if [[ "${var_value}" == "/" || "${var_value}" == "${HOME}" || "${var_value}" == "/root" || "${var_value}" == "/home" ]]; then
            echo "B2: variable \$${var_name}=${var_value} resolves to a protected root path" >&2
            return 1
        fi
    done <<< "${vars}"

    # Literal root usage with destructive commands
    # rm -rf / or rm -rf /*
    if [[ "${first_word}" == "rm" ]]; then
        # Match "rm -rf /" with optional trailing /* and require word boundary
        if echo "$cmd" | grep -qE 'rm[[:space:]]+-[rRfF]+[[:space:]]+/(\*|[[:space:]]|$)'; then
            echo "B2: literal 'rm -rf /' detected" >&2
            return 1
        fi
        # rm -rf $HOME or rm -rf /root
        if echo "$cmd" | grep -qE "rm[[:space:]]+-[rRfF]+[[:space:]]+(/root|/home)(/\*|/[[:space:]]|[[:space:]]|$)"; then
            echo "B2: literal 'rm -rf /root' or '/home' detected" >&2
            return 1
        fi
    fi

    return 0
}

export -f check_bash_path_safety
