#!/bin/bash
# NPU Model Inference Optimization Environment Installer

show_help() {
    cat <<'EOF'
NPU Model Inference Optimization Environment Installer

Usage: bash init.sh [level] [tool]

Arguments:
  level   - "project" (default) or "global"
  tool    - "opencode" (default), "claude", "trae", or "cursor"
  --help  - Show this help message

Examples:
  bash init.sh                      # Project-level, OpenCode
  bash init.sh project opencode     # Project-level, OpenCode
  bash init.sh project claude       # Project-level, Claude Code
  bash init.sh project trae         # Project-level, Trae
  bash init.sh project cursor       # Project-level, Cursor
  bash init.sh global claude        # Global, Claude Code
  bash init.sh global cursor        # Global, Cursor

Installation paths:
  OpenCode project: .opencode/skills/ + .opencode/agents/ + .opencode/AGENTS.md
  Claude  project:  .claude/skills/  + .claude/agents/  + .claude/hooks/ + .claude/CLAUDE.md
  Trae    project:  .trae/skills/    + .trae/agents/    + .trae/AGENTS.md
  Cursor  project:  .cursor/skills/  + .cursor/agents/  + .cursor/AGENTS.md
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LEVEL="project"
TOOL="opencode"

for arg in "$@"; do
    case "$arg" in
        --help) show_help; exit 0 ;;
        global|project) LEVEL="$arg" ;;
        opencode|claude|trae|cursor) TOOL="$arg" ;;
        *) echo "Error: unknown argument '$arg'"; show_help; exit 1 ;;
    esac
done

if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_DIR="$HOME/.config/opencode"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_DIR="$HOME/.cursor"
    elif [ "$TOOL" = "trae" ]; then
        echo "Error: Global installation is not supported for Trae. Use project-level instead."
        exit 1
    else
        CONFIG_DIR="$HOME/.claude"
    fi
else
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_DIR=".opencode"
    elif [ "$TOOL" = "trae" ]; then
        CONFIG_DIR=".trae"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_DIR=".cursor"
    else
        CONFIG_DIR=".claude"
    fi
fi

SKILLS_DIR="$CONFIG_DIR/skills"
AGENTS_DIR="$CONFIG_DIR/agents"

echo "Installing NPU Model Inference Optimization Environment..."
echo "  Level: $LEVEL"
echo "  Tool: $TOOL"
echo "  Skills: $SKILLS_DIR"
echo "  Agents: $AGENTS_DIR"
echo ""

# 1. Install skills
SOURCE_SKILLS_DIR="$MODEL_ROOT/skills"
echo "Installing skills..."
if [ -d "$SOURCE_SKILLS_DIR" ]; then
    mkdir -p "$SKILLS_DIR"
    skill_count=0
    for skill_dir in "$SOURCE_SKILLS_DIR"/model-infer-*; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            ln -sfn "$(realpath "$skill_dir")" "$SKILLS_DIR/$skill_name"
            echo "  Linked: $skill_name"
            skill_count=$((skill_count + 1))
        fi
    done
    echo "  Total: $skill_count skills"
else
    echo "  Warning: $SOURCE_SKILLS_DIR not found"
fi

# 1b. Install workflow skill from team directory
WORKFLOW_SKILL="$SCRIPT_DIR/model-infer-optimize"
if [ -d "$WORKFLOW_SKILL" ]; then
    mkdir -p "$SKILLS_DIR"
    ln -sfn "$(realpath "$WORKFLOW_SKILL")" "$SKILLS_DIR/model-infer-optimize"
    echo "  Linked: model-infer-optimize (workflow)"
fi

# 2. Install agents
SOURCE_AGENTS_DIR="$MODEL_ROOT/agents"
echo ""
echo "Installing agents..."
if [ -d "$SOURCE_AGENTS_DIR" ]; then
    mkdir -p "$AGENTS_DIR"
    agent_count=0
    for agent_file in "$SOURCE_AGENTS_DIR"/model-infer-*.md; do
        if [ -f "$agent_file" ]; then
            agent_name=$(basename "$agent_file")
            ln -sfn "$(realpath "$agent_file")" "$AGENTS_DIR/$agent_name"
            echo "  Linked: ${agent_name%.md}"
            agent_count=$((agent_count + 1))
        elif [ -d "${agent_file%.md}" ]; then
            # Fallback: support directory-based agents (dir/AGENT.md)
            agent_dir="${agent_file%.md}"
            agent_name=$(basename "$agent_dir")
            if [ -f "$agent_dir/AGENT.md" ]; then
                ln -sfn "$(realpath "$agent_dir/AGENT.md")" "$AGENTS_DIR/${agent_name}.md"
                echo "  Linked: $agent_name"
                agent_count=$((agent_count + 1))
            fi
        fi
    done
    echo "  Total: $agent_count agents"
else
    echo "  Warning: $SOURCE_AGENTS_DIR not found"
fi

# 3. Install config file
echo ""
mkdir -p "$CONFIG_DIR"
if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
    ln -sf "$(realpath "$SCRIPT_DIR/AGENTS.md")" "$CONFIG_DIR/AGENTS.md"
    echo "Installed: AGENTS.md"
else
    ln -sf "$(realpath "$SCRIPT_DIR/AGENTS.md")" "$CONFIG_DIR/CLAUDE.md"
    echo "Installed: CLAUDE.md"
fi

# 4. Install hooks (Claude Code only)
SOURCE_HOOKS_DIR="$SCRIPT_DIR/hooks"
if { [ "$TOOL" = "claude" ] || [ "$TOOL" = "cursor" ]; } && [ -d "$SOURCE_HOOKS_DIR" ]; then
    HOOKS_DIR="$CONFIG_DIR/hooks"
    mkdir -p "$HOOKS_DIR"
    echo ""
    echo "Installing hooks..."
    for hook_file in "$SOURCE_HOOKS_DIR"/*.py; do
        if [ -f "$hook_file" ]; then
            hook_name=$(basename "$hook_file")
            ln -sf "$(realpath "$hook_file")" "$HOOKS_DIR/$hook_name"
            echo "  Linked: $hook_name"
        fi
    done

    # Install settings.json with absolute hook paths (works for both project and global installs)
    HOOKS_ABS_DIR="$(cd "$SCRIPT_DIR/hooks" && pwd)"
    if [ ! -f "$CONFIG_DIR/settings.json" ]; then
        cat > "$CONFIG_DIR/settings.json" << EOF
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{"type": "command", "command": "python3 $HOOKS_ABS_DIR/pre_tool_use.py"}]
      },
      {
        "matcher": "Edit|Write|Bash",
        "hooks": [{"type": "command", "command": "python3 $HOOKS_ABS_DIR/time_reminder.py"}]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Read",
        "hooks": [{"type": "command", "command": "python3 $HOOKS_ABS_DIR/post_tool_use.py"}]
      }
    ],
    "SubagentStop": [
      {
        "matcher": "model-infer-implementer|model-infer-reviewer",
        "hooks": [{"type": "command", "command": "python3 $HOOKS_ABS_DIR/subagent_stop.py"}]
      }
    ]
  }
}
EOF
        echo "  Created: settings.json"
    else
        echo "  Skipped: settings.json already exists"
    fi
fi

# 5. Clone reference repo
echo ""
echo "Setting up reference repo..."
REFERENCE_DIR="$SCRIPT_DIR/cann-recipes-infer"
if [ -d "$REFERENCE_DIR" ] && [ -d "$REFERENCE_DIR/.git" ]; then
    echo "  Reference repo exists, pulling latest..."
    cd "$REFERENCE_DIR" && git pull --quiet 2>/dev/null && cd "$SCRIPT_DIR"
else
    if command -v git &> /dev/null; then
        echo "  Cloning reference repo..."
        if git clone --depth 1 https://gitcode.com/cann/cann-recipes-infer.git "$REFERENCE_DIR" 2>/dev/null; then
            echo "  Cloned: $REFERENCE_DIR"
        else
            echo "  Warning: Clone failed. Please clone manually:"
            echo "    git clone https://gitcode.com/cann/cann-recipes-infer.git $REFERENCE_DIR"
        fi
    else
        echo "  Warning: git not found. Please install git and clone manually:"
        echo "    git clone https://gitcode.com/cann/cann-recipes-infer.git $REFERENCE_DIR"
    fi
fi

# 6. Health check
echo ""
echo "Health check..."
health_ok=true
for check_dir in "$SKILLS_DIR" "$AGENTS_DIR"; do
    if [ -d "$check_dir" ]; then
        count=$(ls "$check_dir" 2>/dev/null | wc -l)
        echo "  ✓ $(basename "$check_dir"): $count items"
    else
        echo "  ✗ $(basename "$check_dir"): missing"
        health_ok=false
    fi
done
if { [ "$TOOL" = "claude" ] || [ "$TOOL" = "cursor" ]; } && [ -f "$CONFIG_DIR/settings.json" ]; then
    echo "  ✓ hooks + settings.json"
fi

echo ""
echo "Installation complete!"
echo "  Level: $LEVEL"
echo "  Tool: $TOOL"
echo ""
echo "Usage:"
echo "  $TOOL"
