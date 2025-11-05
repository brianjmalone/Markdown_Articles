# Extracting Lines from Files: Use the Right Tool (2025)

You're need lines 2840 through 2860—just the error and its immediate context from a massive log file. 

So, ```sed```, right? 

Right?

Maybe. Turns out, sometimes you should use `sed`. Sometimes you shouldn't. We've got cool new tools now, thanks to Rust. Here's how to know the difference.

**Time investment:** 5-minute read. The examples work in under 30 seconds each.

**Bottom line:** For known line numbers, `bat` is easier. For context around search terms, `ripgrep` is faster. For pattern-to-pattern ranges, `sed` is still the only good option. For complex logic, use Python.

---

## Table of Contents

- [The Decision Tree](#the-decision-tree)
- [Use Case 1: Known Line Numbers → bat](#use-case-1-known-line-numbers--bat)
- [Use Case 2: Context Around Matches → ripgrep](#use-case-2-context-around-matches--ripgrep)
- [Use Case 3: Pattern-to-Pattern Ranges → sed](#use-case-3-pattern-to-pattern-ranges--sed)
- [Use Case 4: Complex Logic → Python](#use-case-4-complex-logic--python)
- [The bash vs Python Boundary](#the-bash-vs-python-boundary)
- [When Things Break](#when-things-break)
- [What I Actually Do](#what-i-actually-do)
- [Next Steps](#next-steps)

---

## The Decision Tree

**Start here. Pick your scenario, skip to that section.**

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

**Tools I don't install:**
- `sd`, `amber`, other sed alternatives — solving problems I don't have
- Complex awk scripts — if I need that much logic, I use Python

---

## Next Steps

**If you're comfortable with line extraction, the next level is field extraction and transformation.**

That's where `awk` enters the picture. `awk` is to columns what `sed` is to lines—a specialized tool for a specific job.

Quick taste of what `awk` does:
```bash
# Print the third column of CSV data
awk -F',' '{print $3}' data.csv

# Sum all numbers in the second column
awk '{sum += $2} END {print sum}' numbers.txt
```

But that's a whole other article. For now, you've got line extraction covered.

**Related topics worth exploring:**
- Stream editing and find/replace → `sd` (modern sed alternative)
- Advanced text processing → `awk` (field-based processing)
- Full codebase searching → `ripgrep` deep dive

---

## Decision Framework Summary

| Scenario | Tool | Why |
|----------|------|-----|
| Lines 340-380 | `bat -r 340:380` | Clearest syntax, best output |
| 5 lines around "ERROR" | `rg -C 5 "ERROR"` | Fast, simple context |
| From pattern to pattern | `sed -n '/start/,/end/p'` | Only tool that does this well |
| Complex logic | Python | Explicit state, readable |
| Just viewing a file | `bat` or `less` | Syntax highlighting, paging |
| File under 100 lines | Your editor | Don't over-optimize |

**The real skill isn't mastering `sed`.** The real skill is knowing when to use `sed`, when to use `bat`, when to use `ripgrep`, and when to just write 5 lines of Python and move on.

Stop tool-shopping. Pick the right hammer. Get back to hammering.

---

*Last updated: January 2025*
