# Text Processing in 2025: A Tool Landscape

You need lines 2840 through 2860 from a massive log file. So, `sed`, right?

Maybe. Or `bat`. Or `ripgrep`. Or Python.

The Unix text processing toolkit has evolved. Rust rewrote half of it with better UX and performance. Meanwhile, sed and awk are still here, doing what they've always done.

This guide goes **deep on line extraction** (that's the most common task) and surveys the rest of the text processing landscape: find/replace, field processing, and when to bail to Python.

**Time investment:** 10-minute read for the full tour. The line extraction section alone is 5 minutes.

**Structure:** Deep dive on line extraction, brief survey of other text processing tasks, decision frameworks throughout.

---

## Table of Contents

**Part 1: Line Extraction (Deep Dive)**
- [Line Extraction Decision Tree](#line-extraction-decision-tree)
- [Known Line Numbers → bat](#use-case-1-known-line-numbers--bat)
- [Context Around Matches → ripgrep](#use-case-2-context-around-matches--ripgrep)
- [Pattern-to-Pattern Ranges → sed](#use-case-3-pattern-to-pattern-ranges--sed)
- [Complex Logic → Python](#use-case-4-complex-logic--python)
- [When Line Extraction Breaks](#when-things-break)

**Part 2: Other Text Processing (Survey)**
- [Find & Replace](#find--replace)
- [Field Processing](#field-processing)
- [The bash vs Python Boundary](#the-bash-vs-python-boundary)

**Practical Guidance**
- [What I Actually Use](#what-i-actually-use)
- [Decision Framework Summary](#decision-framework-summary)

---

# Part 1: Line Extraction (Deep Dive)

## Line Extraction Decision Tree

**Start here for line extraction. Pick your scenario, skip to that section.**

- **I know the exact line numbers** (e.g., lines 340-380) → [Use bat](#use-case-1-known-line-numbers--bat)
- **I need lines around a search term** (e.g., 5 lines before/after "ERROR") → [Use ripgrep](#use-case-2-context-around-matches--ripgrep)
- **I need everything between two patterns** (e.g., start of function to closing brace) → [Use sed](#use-case-3-pattern-to-pattern-ranges--sed)
- **I need custom logic or state tracking** (e.g., "find the *second* occurrence") → [Use Python](#use-case-4-complex-logic--python)

**If your file is under 100 lines, just open it in your editor.** Seriously. Don't optimize a non-problem.

---

## Use Case 1: Known Line Numbers → bat

**Scenario:** Your error message says "line 2847" or you ran `grep -n` and know exactly which lines you want.

### The Modern Approach

```bash
bat --line-range 340:380 logfile.txt
```

**What you get:**
- Syntax highlighting (if it's code)
- Line numbers by default
- Git integration (shows modifications)
- Automatic paging for long output
- Clean, readable output

### Variations

```bash
# Single line
bat -r 350 file.txt

# From line 340 to end
bat -r 340: file.txt

# First 40 lines
bat -r :40 file.txt

# Relative range (340 plus next 40 lines)
bat -r 340:+40 file.txt
```

**Installation:**
```bash
brew install bat        # macOS
sudo apt install bat    # Ubuntu/Debian
```

### The Old Way (sed)

```bash
sed -n '340,380p' logfile.txt
```

**Why I don't use this anymore for line numbers:**
- `bat`'s syntax is clearer: `340:380` vs `'340,380p'`
- No `-n` flag to remember
- Better default output (line numbers, highlighting)
- Consistent with how I think: "lines 340 to 380"

**When to still use sed:** If `bat` isn't installed and you can't install it. Or if you're piping output to another command and don't want the formatting.

### head/tail Combo (Don't Do This)

```bash
# Please don't
head -n 380 file.txt | tail -n 41
```

This makes you do math (380 - 340 + 1 = 41) and reads more of the file than necessary. Just use `bat`.

---

## Use Case 2: Context Around Matches → ripgrep

**Scenario:** You're searching for "ERROR" and need to see what happened right before and after.

### The Fast Approach

```bash
rg -A 5 -B 5 "ERROR" logfile.txt
```

**What you get:**
- 5 lines before each match
- The matching line itself
- 5 lines after each match
- Blazingly fast (faster than grep)
- Respects .gitignore by default

### Variations

```bash
# 10 lines of context (before AND after)
rg -C 10 "ERROR" logfile.txt

# Just lines after
rg -A 5 "ERROR" logfile.txt

# Just lines before
rg -B 5 "ERROR" logfile.txt

# Case insensitive
rg -i -C 5 "error" logfile.txt
```

**Installation:**
```bash
brew install ripgrep    # macOS
sudo apt install ripgrep  # Ubuntu/Debian
```

### Why Not grep?

`grep` has the same flags (`-A`, `-B`, `-C`), but `ripgrep` is faster and has better defaults:

```bash
# grep works, but is slower on large codebases
grep -A 5 -B 5 "ERROR" logfile.txt
```

If `ripgrep` isn't installed, `grep` is fine. The syntax is identical.

### What ripgrep Can't Do

**ripgrep cannot extract ranges between two patterns.**

This doesn't work:
```bash
# NOT POSSIBLE with ripgrep
rg '/start_pattern/,/end_pattern/' file.txt  # ❌
```

There's an open feature request for this, but as of 2025, it doesn't exist. For pattern-to-pattern ranges, you need `sed`.

---

## Use Case 3: Pattern-to-Pattern Ranges → sed

**Scenario:** You need everything from the start of a function to its closing brace. Or from one timestamp to another. Or from a section header to the next header.

**This is where `sed` still wins.**

### The Code

```bash
sed -n '/start_pattern/,/end_pattern/p' filename
```

### The Code with Annotations

```bash
# -n: Quiet mode (don't print everything)
# /start_pattern/: Start extracting when you see this
# ,: Through...
# /end_pattern/: Until you see this
# p: Print those lines
sed -n '/start_pattern/,/end_pattern/p' filename
```

### Real Examples

**Extract a shell function:**
```bash
sed -n '/^function backup/,/^}/p' .bashrc
```

**Get a config section:**
```bash
sed -n '/\[database\]/,/\[.*\]/p' config.ini
```

**Pull error context from logs (pattern-based):**
```bash
sed -n '/ERROR: Auth failed/,/^[0-9][0-9]:[0-9][0-9]/p' app.log
```
*From the auth error to the next timestamp.*

**Extract commented documentation:**
```bash
sed -n '/# BEGIN: Authentication/,/# END: Authentication/p' script.py
```

### Important Behaviors

**Ranges are inclusive** — both boundary lines are included in the output.

**First match wins** — `sed` starts at the first occurrence of `start_pattern` and stops at the first `end_pattern` after that.

**Greedy by default** — if the end pattern never appears, `sed` prints to the end of the file.

### Mixed Boundaries

You can combine line numbers and patterns:

```bash
# From line 50 to the next pattern
sed -n '50,/^# End of aliases/p' .bashrc

# From pattern to line 100
sed -n '/# Custom functions/,100p' .bashrc

# From pattern to end of file
sed -n '/export PATH/,$p' .bashrc
```

### Why bat and ripgrep Can't Do This

`bat` only accepts line numbers. `ripgrep` only does fixed context (N lines before/after). Neither can say "from *this* pattern until you see *that* pattern."

**This is `sed`'s niche in 2025.** It's not obsolete—it's specialized.

---

## Use Case 4: Complex Logic → Python

**Scenario:** You need something `sed`, `bat`, and `ripgrep` can't do:
- Find the *second* occurrence of a pattern
- Track state between lines
- Apply conditional logic
- Process fields differently based on content

**When the one-liner gets complicated, use Python.**

### Example: Line Range (Python vs bash)

**bash (bat):**
```bash
bat -r 340:380 file.txt
```

**Python:**
```python
with open("file.txt") as f:
    for i, line in enumerate(f, 1):
        if 340 <= i <= 380:
            print(line, end='')
```

For this use case, Python is overkill. Use `bat`.

### Example: Pattern Range (Python vs sed)

**bash (sed):**
```bash
sed -n '/# START/,/# END/p' script.py
```

**Python:**
```python
printing = False
with open("script.py") as f:
    for line in f:
        if "# START" in line:
            printing = True
        if printing:
            print(line, end='')
        if "# END" in line:
            printing = False
```

Python is more verbose but explicit. The state (`printing = True/False`) is clear. For simple ranges, `sed` wins. For "find the second START" or other stateful logic, Python wins.

### When Python Becomes Worth It

**Find the second occurrence:**
```python
# Can't do this easily in sed
count = 0
with open("file.txt") as f:
    for line in f:
        if "ERROR" in line:
            count += 1
            if count == 2:
                print(line)
                break
```

**Conditional extraction:**
```python
# Extract lines between patterns, but only if they contain "DEBUG"
printing = False
with open("app.log") as f:
    for line in f:
        if "START" in line:
            printing = True
        if printing and "DEBUG" in line:
            print(line, end='')
        if "END" in line:
            printing = False
```

**Field processing with logic:**
```python
# This is where awk or Python shine, not sed
with open("data.csv") as f:
    for line in f:
        fields = line.split(',')
        if len(fields) > 3 and int(fields[2]) > 100:
            print(fields[0], fields[3])
```

> **Best tool = the one you stop thinking about**

If you're spending more than 60 seconds fighting with `sed` syntax, write 5 lines of Python.

---

## The bash vs Python Boundary

**Use bash tools (bat, ripgrep, sed) when:**
- You're working interactively at the command line
- The extraction is simple and well-defined
- You're piping output to another command
- Speed of typing the command matters
- You want the result right now, no script file

**Use Python when:**
- The extraction logic is part of a larger automated workflow
- You need state, counters, or complex conditionals
- You're going to run this more than once (save the script)
- Readability for your team matters more than conciseness
- You need proper error handling or logging
- The bash one-liner exceeds ~80 characters or needs variables

**Real decision I made last week:**

*Task:* Extract all functions from a Python file, but only if they have a docstring.

*Initial thought:* Could I do this with `sed`? Probably, with enough regex pain.

*Actual decision:* Wrote 15 lines of Python in 3 minutes. Worked first try. Moved on.

**Time spent fighting the tool is time wasted.** Pick the tool that gets you back to your actual work fastest.

---

## When Things Break

### Pattern Doesn't Match

```bash
$ sed -n '/FOOBAR/,/BAZQUX/p' file.txt
# (no output)
```

**What happened:** The start pattern wasn't found. `sed` printed nothing.

**Fix:** Verify the pattern exists first:
```bash
grep "FOOBAR" file.txt  # Does this show anything?
```

### End Pattern Never Appears

```bash
$ sed -n '/START/,/END/p' file.txt
# (prints from START to end of file)
```

**What happened:** `sed` found START but never found END, so it printed everything after START.

**Fix:** Check that both patterns exist:
```bash
grep "START" file.txt
grep "END" file.txt
```

### File Doesn't Exist

```bash
$ bat -r 10:20 nonexistent.txt
Error: 'nonexistent.txt': No such file or directory
```

**Fix:** Check the path. Use tab completion. Verify with `ls`.

### Binary File Warning

```bash
$ sed -n '1,10p' image.png
# (garbage output)
```

**What happened:** `sed` is for text files. Don't use it on binaries.

**Fix:** Check file type first:
```bash
file myfile.txt  # Tells you if it's text or binary
```

### File Too Large for Memory

Python scripts that load entire files into memory can fail on multi-GB files:

```python
# DON'T DO THIS on huge files
lines = open("huge.log").readlines()  # ❌ Loads whole file
```

**Fix:** Stream tools (`bat`, `sed`, `ripgrep`) handle large files fine. If using Python, iterate line by line:
```python
# DO THIS instead
with open("huge.log") as f:
    for line in f:  # ✅ Streams one line at a time
        # process line
```

### Regex Special Characters

```bash
$ sed -n '/function()/,/}/p' code.js
sed: 1: "/function()/,/}/p": invalid command code )
```

**What happened:** Parentheses are special in regex. You need to escape them.

**Fix:**
```bash
sed -n '/function()/,/}/p' code.js  # Often works as-is
# or
sed -n '/function\(\)/,/}/p' code.js  # Explicitly escaped
```

When in doubt, test your pattern with `grep` first to verify it matches.

---

# Part 2: Other Text Processing (Survey)

## Find & Replace

### Modern Approach: sd

```bash
sd 'old_text' 'new_text' file.txt
```

**What sd does:** Find and replace with sane syntax. Uses regex from JavaScript/Python (the syntax you already know). 2-11x faster than sed for substitution.

**Installation:**
```bash
brew install sd              # macOS
cargo install sd             # via Rust
```

**Examples:**
```bash
# Simple replacement
sd 'foo' 'bar' file.txt

# Regex with capture groups
sd '(\w+)@(\w+)' '$1@example.com' emails.txt

# In-place edit (like sed -i)
sd 'old' 'new' file.txt

# Preview before replacing
sd 'old' 'new' file.txt --preview
```

**When to use sd:** You're doing find/replace and don't want to remember sed's escaping rules.

---

### Classic Approach: sed

```bash
# Basic substitution
sed 's/old/new/g' file.txt

# In-place editing
sed -i '' 's/old/new/g' file.txt  # macOS
sed -i 's/old/new/g' file.txt     # Linux

# Replace only on lines matching a pattern
sed '/pattern/s/old/new/g' file.txt
```

**When to use sed:** It's everywhere. You already know the syntax. You're doing more than just substitution (combining with line ranges, etc.).

**sed's advantage:** Can combine substitution with line selection. sd can't do this:
```bash
# Replace only in lines 10-20
sed '10,20s/old/new/g' file.txt

# Replace only between patterns
sed '/START/,/END/s/old/new/g' file.txt
```

---

### Interactive Multi-File: amber

```bash
ambr 'search' 'replace'      # Interactive prompts for each match
```

**What it does:** Search and replace across directories with confirmation for each change. Like your editor's "find in files" but command-line.

**Installation:** https://github.com/dalance/amber

**When to use it:** Refactoring across multiple files where you want to review each change. Most people use their editor for this.

---

## Field Processing

### Column Operations: awk

```bash
# Print third column
awk '{print $3}' data.txt

# CSV with custom delimiter
awk -F',' '{print $3}' data.csv

# Filter rows where column 2 > 100
awk '$2 > 100' data.txt

# Sum numbers in second column
awk '{sum += $2} END {print sum}' numbers.txt

# Multiple conditions
awk '$2 > 100 && $3 < 50 {print $1, $2}' data.txt
```

**What awk excels at:**
- Processing tabular/columnar data
- Quick field extraction and filtering
- Simple calculations on columns
- Log file analysis (splitting on whitespace)

**When to use awk:**
- Working with structured text (CSV, logs, tables)
- Simple field operations (print, filter, sum)
- One-liner territory

**When to use Python instead:**
- Logic exceeds a few conditions
- Need proper CSV parsing (quoted fields, escapes)
- Calculations requiring libraries
- Readability for others matters

---

### Python for Structured Data

```python
# Simple field extraction
with open('data.txt') as f:
    for line in f:
        fields = line.split()
        if len(fields) >= 3 and int(fields[1]) > 100:
            print(fields[0], fields[2])
```

```python
# Proper CSV handling
import csv
with open('data.csv') as f:
    for row in csv.DictReader(f):
        if int(row['amount']) > 100:
            print(row['name'], row['total'])
```

**Python wins when:**
- CSV files with quoted fields, embedded commas
- Complex field logic
- Need libraries (datetime parsing, calculations)
- Part of larger script

---

## What I Actually Do

**When I'm exploring a codebase:**
- Quick file preview: `bat filename` (first screenful, syntax highlighted)
- Known line numbers: `bat -r 340:380 filename`
- Search with context: `rg -C 5 "keyword"`

**When I'm debugging:**
- Error logs with context: `rg -A 10 -B 5 "ERROR"`
- Specific log range: `bat -r 2840:2860 app.log`
- Extract between timestamps: `sed -n '/2025-01-15 14:23/,/2025-01-15 14:25/p' app.log`

**When I'm writing automation:**
- Simple extraction in a script: bash tools
- Any conditional logic: Python immediately
- Anything I'll run more than twice: Python script, saved

**Tools I keep installed:**
- `bat` — use it daily
- `ripgrep` — use it constantly
- `sed` — use it weekly for pattern ranges
- Python — when the one-liner fights back

**For find/replace:**
- `sd` for simple substitutions (better UX than sed)
- `sed` when combining with line ranges
- Editor's find/replace for interactive multi-file

**For field processing:**
- `awk` for quick column operations on logs
- Python when CSV gets complex or logic exceeds one-liner

**Tools I don't install:**
- `amber` — editor's find/replace does this better
- Complex awk scripts — if it's complex, I use Python

---

## Decision Framework Summary

**Line Extraction:**

| Scenario | Tool | Why |
|----------|------|-----|
| Lines 340-380 | `bat -r 340:380` | Clearest syntax, best output |
| 5 lines around "ERROR" | `rg -C 5 "ERROR"` | Fast, simple context |
| From pattern to pattern | `sed -n '/start/,/end/p'` | Only tool that does this |
| Complex line logic | Python | Explicit state, readable |

**Find & Replace:**

| Scenario | Tool | Why |
|----------|------|-----|
| Simple substitution | `sd 'old' 'new'` | Better syntax than sed |
| Replace in line range | `sed '10,20s/old/new/g'` | sed can combine ranges |
| Multi-file refactor | Editor find/replace or `ambr` | Interactive review |

**Field Processing:**

| Scenario | Tool | Why |
|----------|------|-----|
| Print column 3 | `awk '{print $3}'` | Built for this |
| Sum column values | `awk '{sum += $2} END {print sum}'` | Simple calculations |
| Complex CSV | Python `csv` module | Proper parsing, readability |
| Field logic with conditions | Python | When awk gets cryptic |

**General:**

| Scenario | Tool | Why |
|----------|------|-----|
| Just viewing a file | `bat` or `less` | Syntax highlighting, paging |
| File under 100 lines | Your editor | Don't over-optimize |
| Bash one-liner > 80 chars | Python | Readability, maintainability |

**The real skill isn't mastering `sed`.** The real skill is knowing when to use `sed`, when to use `bat`, when to use `ripgrep`, and when to just write 5 lines of Python and move on.

Stop tool-shopping. Pick the right hammer. Get back to hammering.

---

*Last updated: January 2025*
