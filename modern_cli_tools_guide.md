# Modern CLI Tools Guide

A comprehensive guide to modern replacements for traditional Unix command-line tools.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Tool #1: bat (replaces cat)](#tool-1-bat-replaces-cat)
4. [Tool #2: eza (replaces ls)](#tool-2-eza-replaces-ls)
5. [Tool #3: ripgrep (replaces grep)](#tool-3-ripgrep-replaces-grep)
6. [Tool #4: fd (replaces find)](#tool-4-fd-replaces-find)
7. [Tool #5: fzf (interactive fuzzy finder)](#tool-5-fzf-interactive-fuzzy-finder)
8. [Custom Functions](#custom-functions)
9. [Complete Setup](#complete-setup)

---

## Overview

### Why Replace Traditional Tools?

Traditional Unix tools (`cat`, `ls`, `find`, `grep`) were designed in the 1970s. Modern replacements offer:

- **10-100x faster performance** (written in Rust/Go)
- **Better defaults** (colors, formatting, sensible options)
- **Git integration** (shows file status)
- **Syntax highlighting** (automatic language detection)
- **Smart filtering** (respects `.gitignore`)

### Comparison Table

| Modern Tool | Replaces | Key Improvement | Language |
|-------------|----------|-----------------|----------|
| `bat` | `cat` | Syntax highlighting, Git integration | Rust |
| `eza` | `ls` | Icons, Git status, tree view | Rust |
| `ripgrep` (`rg`) | `grep` | 10-100x faster, respects `.gitignore` | Rust |
| `fd` | `find` | Simple syntax, fast, colored output | Rust |
| `fzf` | (new concept) | Interactive fuzzy search | Go |

### Decision Tree: Which Tools Should You Install?

**Start here →** How often do you work in the terminal?

**Rarely (< 1 hour/day):**
- Start with: `bat` only
- Why: Instant value, zero learning curve (syntax highlighting is the value-add)

**Regularly (1-3 hours/day):**
- Essential: `bat`, `ripgrep`, `fzf`
- Optional: `eza`, `fd`
- Why: These save the most time for code search/navigation

**Power user (3+ hours/day):**
- Install: Everything + custom functions
- Why: Compounding time savings, improved workflow

**Git-heavy workflows?**
- Add: `eza` (Git status in ls) + `fbr()` function (branch switching)

**Not sure?**
- Use the "Try Before You Commit" section below

---

## Try Before You Commit (2-Minute Test)

Not ready to configure everything? Try these commands first to see if the value is worth the setup time:

```bash
# Install just the basics (no configuration needed)
brew install bat eza ripgrep

# Test bat vs cat
bat ~/.zshrc              # Syntax highlighting, line numbers
cat ~/.zshrc              # Compare with traditional output

# Test eza vs ls
eza --icons               # Icons and colors
ls                        # Compare with traditional output

# Test ripgrep vs grep
rg "alias" ~/.zshrc       # Fast, colored output
grep "alias" ~/.zshrc     # Compare with traditional output
```

(Replace zshrc with bashrc, as needed.)

**If you like what you see**, continue with the full installation and setup below.

---

## Installation

### Platform Compatibility

| Tool | macOS | Linux | Windows |
|------|-------|-------|---------|
| bat | ✅ brew | ✅ apt/dnf/pacman | ✅ scoop/choco |
| eza | ✅ brew | ✅ cargo/manual | ✅ cargo/manual |
| ripgrep | ✅ brew | ✅ apt/dnf/pacman | ✅ scoop/choco |
| fd | ✅ brew | ✅ apt/dnf/pacman | ✅ scoop/choco |
| fzf | ✅ brew | ✅ apt/dnf/pacman | ✅ scoop/choco |

**This guide uses macOS/Homebrew commands.** For Linux/Windows, see the [Additional Resources](#additional-resources) section for official installation instructions.

### Install All Tools

```bash
# Core tools
brew install bat eza ripgrep fd fzf

# Optional: Color theme generator for eza
brew install vivid

# Install Nerd Font for icons (required for eza icons)
brew install --cask font-meslo-lg-nerd-font
```

### Enable fzf Shell Integration

```bash
# This enables Ctrl+R, Ctrl+T, Alt+C shortcuts
$(brew --prefix)/opt/fzf/install
```

### Configure Terminal Font

**For Terminal.app:**
1. Terminal → Settings (Cmd+,)
2. Text tab
3. Change font
4. Search for "MesloLGS NF" or "Meslo"
5. Select "MesloLGS Nerd Font" (or similar with "Nerd Font" in name)
6. Size: 13-14pt

---

## Tool #1: bat (replaces cat)

### What It Does

`bat` is `cat` with syntax highlighting, line numbers, Git integration, and automatic paging.

### Basic Usage

```bash
# View a file
bat filename.py

# View multiple files
bat file1.py file2.py

# Show line numbers explicitly
bat -n filename.py

# Plain output (for piping)
bat -p filename.py

# Show non-printable characters
bat -A filename.txt
```

### Common Use Cases

#### 1. Quick File Preview
```bash
bat README.md
# Renders markdown with syntax highlighting
```

#### 2. Show Specific Lines
```bash
bat -r 10:20 script.py
# Only shows lines 10-20
```

#### 3. Pipeline with Syntax Highlighting
```bash
curl -s https://api.github.com/users/octocat | bat -l json
# Downloads JSON and displays with syntax highlighting
```

#### 4. Check File Encoding Issues
```bash
bat -A file.txt
# Shows tabs (→), spaces (·), line endings
```

### Configuration

```bash
# Create config file
mkdir -p ~/.config/bat
cat > ~/.config/bat/config << 'EOF'
--theme="Dracula"
--style="numbers,changes,grid"
--italic-text=always
EOF
```

### When NOT to Use bat

- **Piping to other commands** - Use `bat -p` or `cat`
- **Very large files (>100MB)** - Use `less` or `cat`

---

## Tool #2: eza (replaces ls)

### What It Does

`eza` is `ls` with icons, Git status, tree view, and better formatting.

### Basic Usage

```bash
# Simple listing with icons
eza --icons

# Detailed listing
eza -lh --icons --git

# Show hidden files
eza -a --icons

# Tree view
eza --tree --level=2 --icons
```

### Common Use Cases

#### 1. Quick Directory Overview
```bash
eza --icons
# Just filenames with icons
```

#### 2. Detailed Listing with Git Status
```bash
eza -lh --icons --git
# Shows file details plus Git status
```

**Git status indicators:**
- `--` = Unmodified
- `M-` = Modified (not staged)
- `-M` = Modified (staged)
- `N-` = New file (not staged)
- `-N` = New file (staged)
- `??` = Untracked

#### 3. Tree View
```bash
eza --tree --level=2 --icons
# Shows directory structure, 2 levels deep
```

#### 4. Sort by Size
```bash
eza -lh --sort=size --reverse --icons
# Largest files first
```

#### 5. Sort by Modified Time
```bash
eza -lh --sort=modified --reverse --icons
# Most recently modified first
```

#### 6. Only Directories
```bash
eza -D --icons
# Shows only directories
```

### Customizing Colors

#### Option 1: Use Defaults (Recommended)
`eza` has good defaults. No configuration needed.

#### Option 2: Use vivid Themes
```bash
# Install vivid
brew install vivid

# See available themes
vivid themes

# Try a theme
export LS_COLORS="$(vivid generate molokai)"
eza --icons -l

# Add to ~/.zshrc if you like it
echo 'export LS_COLORS="$(vivid generate molokai)"' >> ~/.zshrc
```

**Popular themes:**
- `molokai` - Vibrant, high contrast
- `nord` - Cool blue tones
- `solarized-dark` - Classic dark theme
- `solarized-light` - For light terminals
- `ayu` - Modern, clean

#### Option 3: Custom Colors (Advanced)

```bash
# Override specific file types
# Format: filetype=color_code
# di=directory, ex=executable, ln=symlink, *.py=Python files

# Example: Keep blue/green, remove gold/yellow
export LS_COLORS="di=01;34:ln=01;36:ex=00;32:*.py=00"
#                 ^directories ^symlinks ^executables ^Python
#                 bold blue    bold cyan normal green normal
```

**Color codes:**
- `30-37` = colors (black through white)
- `90-97` = bright colors
- `01` = bold
- `00` = normal

**Common colors:**
- `32` = green
- `34` = blue
- `36` = cyan
- `37` = white

#### Remove Color Customization
```bash
# In current session
unset LS_COLORS

# Remove from ~/.zshrc
# Delete or comment out any LS_COLORS lines
```

### When NOT to Use eza

- **Shell scripts meant for portability** - Use `ls` (available everywhere)
- **CI/CD pipelines** - Use `ls` (standard across systems)
- **Parsing output programmatically** - Use `ls` with specific flags for consistent format

### Icons Not Showing?

If you see `?` boxes instead of icons:

1. **Install a Nerd Font** (see Installation section)
2. **Configure your terminal** to use the Nerd Font
3. **Restart terminal completely**

Or disable icons:
```bash
alias ls='eza'
alias ll='eza -lh --git'
```

---

## Tool #3: ripgrep (replaces grep)

### What It Does

`ripgrep` (`rg`) is 10-100x faster than `grep`, respects `.gitignore`, and has better defaults.

### Basic Usage

```bash
# Basic search
rg "pattern"

# Case-insensitive
rg -i "error"

# Search specific file types
rg "import" --type py

# Show only filenames
rg -l "matplotlib"

# Search with context
rg -C 3 "def train_model"
```

### Common Use Cases

#### 1. Find All TODOs
```bash
rg "TODO"
# Automatically skips .git, __pycache__, etc.
```

#### 2. Search Specific File Types
```bash
rg "import" --type py      # Only Python files
rg "const" --type js       # Only JavaScript files
rg "SELECT" --type sql     # Only SQL files

# See all available types
rg --type-list
```

#### 3. Case-Insensitive Search
```bash
rg -i "error"
# Matches "error", "Error", "ERROR"
```

#### 4. Show Only Filenames
```bash
rg -l "matplotlib"
# Lists files containing "matplotlib"
```

#### 5. Search with Context
```bash
rg -C 3 "def train_model"
# Shows 3 lines before and after match

rg -A 2 "error"  # 2 lines after
rg -B 2 "error"  # 2 lines before
```

#### 6. Exclude Patterns
```bash
rg "error" --glob '!test_*'
# Search but exclude test files
```

#### 7. Count Matches
```bash
rg -c "import pandas"
# Shows count per file
```

#### 8. Replace Preview
```bash
rg "old_name" --replace "new_name"
# Shows what replacement would look like
# Doesn't modify files (just preview)
```

#### 9. Search Hidden Files
```bash
rg "pattern" --hidden
# Includes hidden files
```

#### 10. Search Ignored Files
```bash
rg "pattern" --no-ignore
# Includes files in .gitignore
```

### When NOT to Use rg

- **Need to search ignored files regularly** - Use `rg --no-ignore`
- **Very complex regex** - Some advanced features missing

---

## Tool #4: fd (replaces find)

### What It Does

`fd` is a fast, simple alternative to `find` with colored output and `.gitignore` awareness.

### Basic Usage

```bash
# Find by extension
fd -e py

# Find by pattern
fd "test_"

# Find in specific directory
fd "\.csv$" ~/Desktop/data

# Execute command on results
fd -e py -x wc -l
```

### Common Use Cases

#### 1. Find by Extension
```bash
fd -e py              # All Python files
fd -e csv             # All CSV files
fd -e jpg -e png      # All images
```

#### 2. Find by Pattern
```bash
fd "test_"            # Files/folders starting with "test_"
fd "analysis"         # Anything containing "analysis"
```

#### 3. Find in Specific Directory
```bash
fd "\.csv$" ~/Desktop/data
# Search only in specific location
```

#### 4. Execute Command on Results
```bash
fd -e py -x wc -l
# Count lines in all Python files
# -x = execute command on each result
```

#### 5. Find and Delete
```bash
fd "\.pyc$" -x rm
# Delete all .pyc files
```

#### 6. Find Recently Modified Files
```bash
fd --changed-within 7d   # Last 7 days
fd --changed-within 2h   # Last 2 hours
```

#### 7. Find Large Files
```bash
fd -S +100m
# Files larger than 100 MB
```

#### 8. Only Directories
```bash
fd -t d "data"
# Only directories matching "data"
# -t d = type directory
# -t f = type file
# -t l = type symlink
```

#### 9. Include Hidden Files
```bash
fd -H "config"
# -H = include hidden files
```

#### 10. Show Full Paths
```bash
fd -e py -a
# -a = absolute paths
```

### When NOT to Use fd

- **Complex find expressions** - Use `find` (more mature, handles edge cases)
- **Shell scripts for production** - Use `find` (POSIX standard, portable)
- **Systems where you can't install tools** - Use `find` (built-in)

---

## Tool #5: fzf (Interactive Fuzzy Finder)

### What It Is

`fzf` is a completely new concept: it turns any list into an interactive, searchable menu.

**Think of it as:**
- Ctrl+F for the terminal
- Spotlight search for command-line
- Interactive filter for any command output

### How Fuzzy Matching Works

You don't need exact strings:

```
Files: Python_FM_Age_Analysis, machine_learning_project

Type: "pymlag" → Matches Python_FM_Age_Analysis
Type: "mlp"    → Matches machine_learning_project
```

### Built-in Shell Shortcuts

After running `$(brew --prefix)/opt/fzf/install`, you get three keyboard shortcuts:

#### 1. Ctrl+R - Command History Search

**Old way:**
```bash
# Press up arrow 50 times to find that command
↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ...
```

**New way:**
```bash
<Ctrl+R>
# Type a few letters
# Interactive search appears
# Select with Enter
```

**Usage:**
1. Press `Ctrl+R`
2. Type partial command (e.g., "conda activate")
3. Use `↑/↓` or `Ctrl+J/K` to navigate
4. Press `Enter` - command is pasted
5. Press `Esc` to cancel

#### 2. Ctrl+T - File Search

**Old way:**
```bash
ls data/
ls data/raw/
ls data/raw/experiments/
# Finally find the file...
python script.py data/raw/experiments/trial_005.csv
```

**New way:**
```bash
python script.py <Ctrl+T>
# Interactive search appears
# Type "trial 005"
# Select file, path is pasted
```

**Usage:**
1. Start typing command: `bat `
2. Press `Ctrl+T`
3. Type to filter (e.g., "experiment")
4. `↑/↓` to select
5. Press `Enter` - path is pasted
6. `Tab` to mark multiple files

#### 3. Alt+C - Directory Navigation

**Old way:**
```bash
cd Desktop/
cd Recanzone_Projects/
cd Python_FM_Age_Analysis/
cd data/
```

**New way:**
```bash
<Alt+C>
# Type "pyth age data"
# Select directory
# Automatically cd's there
```

### Keys Inside fzf

- `↑/↓` or `Ctrl+J/K` - Navigate
- `Enter` - Select
- `Tab` - Mark multiple items (multi-select mode)
- `Ctrl+/` - Toggle preview (if configured)
- `Ctrl+C` or `Esc` - Cancel

### Using fzf in Custom Commands

Pipe any output to `fzf`:

#### Example 1: Kill Process
```bash
ps aux | fzf | awk '{print $2}' | xargs kill
```

#### Example 2: Git Branch Checkout
```bash
git branch | fzf | xargs git checkout
```

#### Example 3: View Multiple Files
```bash
fd -e py | fzf -m | xargs bat
# -m enables multi-select (Tab to mark)
```

### fzf Preview Windows

Show live previews as you navigate:

```bash
ls | fzf --preview 'bat --color=always {}'
#                   ^                  ^
#                   Preview command    Placeholder for item
```

**Preview window options:**
```bash
--preview-window=right:60%    # Right side, 60% width
--preview-window=up:40%       # Top, 40% height
--preview-window=down:50%     # Bottom, 50% height
--preview-window=hidden       # Start hidden (Ctrl+/ toggles)
```

### fzf Options

```bash
# Multi-select mode
fzf -m

# Start with a query
fzf --query "test"

# Exact match (no fuzzy)
fzf --exact

# Custom key bindings
fzf --bind 'ctrl-o:execute(bat {})'
```

---

## Custom Functions

Add these to `~/.zshrc` for powerful workflows.

### Function 1: fe() - Find and Edit File

**Purpose:** Quickly find and open files in your editor with live preview.

**When to use:**
- Can't remember exact filename
- Want to preview before opening
- Tired of `cd`-ing around

```bash
fe() {
  local file
  file=$(fzf --preview 'bat --color=always --style=numbers {}' \
             --preview-window=right:60%)
  [ -n "$file" ] && ${EDITOR:-code} "$file"
}
```

**How it works:**
1. Searches all files from current directory (recursively)
2. Shows live preview with syntax highlighting as you navigate
3. Select file with Enter
4. Opens in your editor (uses `$EDITOR` environment variable)

**Usage:**
```bash
fe
# Type to filter (e.g., "analysis py")
# ↑/↓ to navigate
# Watch preview update
# Enter to open
# Esc to cancel
```

**Set your editor:**
```bash
# Add to ~/.zshrc
export EDITOR="code"    # VS Code
# or
export EDITOR="vim"     # Vim
```

### Function 2: fd_cd() - Find and CD to Directory

**Purpose:** Navigate to directories with preview of contents.

**When to use:**
- Don't remember exact path
- Want to see what's inside before going there
- Quick directory exploration

```bash
fd_cd() {
  local dir
  dir=$(find ${1:-.} -maxdepth 5 -type d 2> /dev/null | \
        fzf --preview 'eza --tree --level=1 --icons {}' \
            --preview-window=right:50%)
  if [ -n "$dir" ]; then
    cd "$dir" && eza --icons
  fi
}
```

**How it works:**
1. Searches directories from current location (or specified location)
2. Shows preview of directory contents (tree view with icons)
3. Select directory with Enter
4. Automatically cd's there and shows contents

**Usage:**
```bash
fd_cd                    # Search from current directory
fd_cd ~/Desktop          # Search from Desktop
fd_cd /                  # Search entire filesystem
```

```bash
# What you do:
# Type to filter (e.g., "anal data")
# ↑/↓ to navigate
# Watch preview show directory contents
# Enter to cd there
# Esc to cancel
```

### Function 3: fbr() - Fuzzy Git Branch Checkout

**Purpose:** Switch Git branches with preview of recent commits.

**When to use:**
- Working with many branches
- Can't remember exact branch name
- Want to see commit history before switching

```bash
fbr() {
  local branches branch
  branches=$(git branch -a | grep -v HEAD) &&
  branch=$(echo "$branches" | 
    fzf --preview 'git log --color=always --oneline --graph --date=short \
                   --pretty="format:%C(auto)%cd %h%d %s" \
                   $(echo {} | sed "s/.* //" | sed "s#remotes/[^/]*/##") | head -50' \
        --preview-window=right:60%) &&
  git checkout $(echo "$branch" | sed "s/.* //" | sed "s#remotes/[^/]*/##")
}
```

**How it works:**
1. Lists all Git branches (local + remote)
2. Shows preview of recent commits (with graph) for each branch
3. Select branch with Enter
4. Automatically checks out the branch

**Usage:**
```bash
fbr
# Must be in a Git repository

# Type to filter (e.g., "feature")
# ↑/↓ to navigate
# Watch preview show commit history
# Enter to checkout branch
# Esc to cancel
```

**After Enter:**
```
Switched to branch 'feature/preprocessing'
```

---

## Complete Setup

### Step 1: Install Everything

If you haven't already, follow the [Installation](#installation) section above to install all tools.

### Step 2: Configure Terminal Font

Set terminal font to "MesloLGS Nerd Font" (size 13-14).

### Step 3: Add to ~/.zshrc

```bash
# Open config file
code ~/.zshrc
# or
nano ~/.zshrc
```

Add this configuration:

```bash
# ============================================
# Modern CLI Tools Configuration
# ============================================

# Editor preference
export EDITOR="code"

# Optional: Color theme for eza
# export LS_COLORS="$(vivid generate molokai)"

# ============================================
# Aliases
# ============================================
# Note: These aliases replace traditional commands (cat → bat, ls → eza).
# Alternative: Use unique names to preserve muscle memory:
#   alias c='bat'
#   alias l='eza --icons'
# Choose what works for your workflow.

# bat (better cat)
alias cat='bat'

# eza (better ls)
alias ls='eza --icons'
alias ll='eza -lh --icons --git'
alias la='eza -lah --icons --git'
alias lt='eza --tree --level=2 --icons'
alias tree='eza --tree --icons'

# ripgrep shortcuts
alias rgf='rg --files-with-matches'

# ============================================
# Custom Functions
# ============================================

# fe - Find and edit file
fe() {
  local file
  file=$(fzf --preview 'bat --color=always --style=numbers {}' \
             --preview-window=right:60%)
  [ -n "$file" ] && ${EDITOR:-code} "$file"
}

# fd_cd - Find and cd to directory
fd_cd() {
  local dir
  dir=$(find ${1:-.} -maxdepth 5 -type d 2> /dev/null | \
        fzf --preview 'eza --tree --level=1 --icons {}' \
            --preview-window=right:50%)
  if [ -n "$dir" ]; then
    cd "$dir" && eza --icons
  fi
}

# fbr - Fuzzy Git branch checkout
fbr() {
  local branches branch
  branches=$(git branch -a | grep -v HEAD) &&
  branch=$(echo "$branches" | 
    fzf --preview 'git log --color=always --oneline --graph --date=short \
                   --pretty="format:%C(auto)%cd %h%d %s" \
                   $(echo {} | sed "s/.* //" | sed "s#remotes/[^/]*/##") | head -50' \
        --preview-window=right:60%) &&
  git checkout $(echo "$branch" | sed "s/.* //" | sed "s#remotes/[^/]*/##")
}
```

### Step 4: Reload Configuration

```bash
source ~/.zshrc
```

### Step 5: Test Everything

```bash
# Test eza
ll

# Test bat
bat ~/.zshrc

# Test ripgrep
rg "alias" ~/.zshrc

# Test fd
fd -e md

# Test fzf shortcuts
# Press Ctrl+R (command history)
# Press Ctrl+T (file search)
# Press Alt+C (directory search)

# Test custom functions
fe        # Find and edit
fd_cd     # Navigate
fbr       # Git branches (in a Git repo)
```

---

## Quick Reference

### Daily Commands

```bash
# Better ls
ll                    # Detailed list with Git status
la                    # Include hidden files
lt                    # Tree view

# Better cat
bat filename.py       # View with syntax highlighting

# Fast search
rg "pattern"          # Search code
fd -e py              # Find Python files

# Interactive
fe                    # Find and edit file
fd_cd                 # Navigate to directory
fbr                   # Switch Git branch
```

### Keyboard Shortcuts

```bash
Ctrl+R                # Search command history
Ctrl+T                # Insert file path
Alt+C                 # Jump to directory
```

### fzf Keys (Inside Interactive Mode)

```bash
↑/↓                   # Navigate
Ctrl+J/K              # Navigate (alternative)
Enter                 # Select
Tab                   # Mark multiple
Ctrl+/                # Toggle preview
Esc                   # Cancel
```

---

## Troubleshooting

### Icons Show as ?

1. Install Nerd Font: `brew install --cask font-meslo-lg-nerd-font`
2. Set terminal font to "MesloLGS Nerd Font"
3. Restart terminal completely
4. Or disable icons: `alias ls='eza'`

### Colors Look Bad

```bash
# Remove color customization
unset LS_COLORS

# Remove from ~/.zshrc
# Delete or comment out LS_COLORS lines

# Reload
source ~/.zshrc
```

### fzf Shortcuts Don't Work

```bash
# Run fzf installer
$(brew --prefix)/opt/fzf/install

# Restart terminal
# Try shortcuts: Ctrl+R, Ctrl+T, Alt+C
```

### Custom Functions Don't Work

```bash
# Check if functions are in ~/.zshrc
grep "fe()" ~/.zshrc

# Reload config
source ~/.zshrc

# Test
type fe
```

### Want to Uninstall?

```bash
# Remove tools
brew uninstall bat eza ripgrep fd fzf vivid

# Remove aliases/functions from ~/.zshrc (delete the configuration block)
# Reload: source ~/.zshrc
```

---

## Additional Resources

- **bat:** https://github.com/sharkdp/bat
- **eza:** https://github.com/eza-community/eza
- **ripgrep:** https://github.com/BurntSushi/ripgrep
- **fd:** https://github.com/sharkdp/fd
- **fzf:** https://github.com/junegunn/fzf
- **vivid:** https://github.com/sharkdp/vivid
- **Nerd Fonts:** https://www.nerdfonts.com

---

## Design Patterns Demonstrated

These tools follow the **Unix Philosophy:**
- Do one thing well
- Compose with other tools
- Text-based input/output

**Performance Benefits:**
- Native tools (Rust/Go) are 10-100x faster than traditional GNU utilities
- Parallel processing
- Smart defaults (skip ignored files)

**Modern UX Principles:**
- Sensible defaults
- Colored output
- Clear error messages
- Interactive when helpful