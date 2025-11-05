# Bash Substitutions & Expansions Cheat Sheet

## Table of Contents

- [Command Substitution](#command-substitution)
- [History Expansion](#history-expansion)
- [Variable Substitution with Defaults](#variable-substitution-with-defaults)
- [String Prefix/Suffix Removal](#string-prefixsuffix-removal)
- [String Replacement](#string-replacement)
- [Substring Extraction](#substring-extraction)
- [String Length](#string-length)
- [Arithmetic Expansion](#arithmetic-expansion)
- [Brace Expansion](#brace-expansion)
- [Array Expansion](#array-expansion)
- [Process Substitution](#process-substitution)
- [Tilde Expansion](#tilde-expansion)
- [Parameter Expansion - Case Modification](#parameter-expansion---case-modification)
- [Indirect Expansion](#indirect-expansion)
- [Quick Reference Summary](#quick-reference-summary)
- [Common Gotchas](#common-gotchas)

---

## Command Substitution

### Use Case: Capture command output as a variable

```bash
current_date=$(date +%Y-%m-%d)
file_count=$(ls -1 | wc -l)
echo "Today is $current_date and we have $file_count files"
```

```bash
# $(...) executes the command inside and substitutes its output
current_date=$(date +%Y-%m-%d)

# Another command substitution - counts lines from ls
file_count=$(ls -1 | wc -l)

# Variables are then used normally with $var
echo "Today is $current_date and we have $file_count files"

# Note: $(command) is preferred over backticks `command`
# Backticks are older syntax and harder to nest
```

---

## History Expansion

### Use Case: Reuse previous commands and their output

```bash
date
result=$(!!)
echo "Captured: $result"
```

```bash
# !! - Refers to the previous command
# $(!!) - Captures the OUTPUT of re-running the previous command
date

# Pro-tip: Verify the command works first, then capture its output
result=$(!!)  # Re-runs 'date' and captures output

echo "Captured: $result"

# IMPORTANT: $(!!) RE-RUNS the command, doesn't use cached output
# For truly identical output, capture initially: result=$(date)
```

### More history expansion tricks

```bash
mkdir /some/long/path/to/directory
cd !$
grep pattern file1.txt file2.txt
wc -l !*
echo one two three
echo "Second arg was: !:2"
```

```bash
# !$ - Last argument of previous command
mkdir /some/long/path/to/directory
cd !$  # Expands to: cd /some/long/path/to/directory

# !* - All arguments from previous command (excluding command itself)
grep pattern file1.txt file2.txt file3.txt
wc -l !*  # Gets: file1.txt file2.txt file3.txt

# !:n - Nth argument from previous command (1-indexed)
echo one two three
echo "Second arg was: !:2"  # "two"

# !:0 - Command name itself
# !:^ - First argument (same as !:1)
# !:$ - Last argument (same as !$)

# Other useful forms:
# !pattern - Most recent command starting with 'pattern'
# !?pattern - Most recent command containing 'pattern'
# ^old^new - Replace 'old' with 'new' in previous command

# History expansion happens BEFORE other expansions
# May be disabled in scripts - check with: set -H (enable) / set +H (disable)
```

---

## Variable Substitution with Defaults

### Use Case: Provide fallback values for unset/empty variables

```bash
username=${USER:-"guest"}
config_file=${CONFIG_PATH:-"/etc/default.conf"}
echo "Running as: $username with config: $config_file"
```

```bash
# ${var:-default} - If var is unset or empty, use "default"
# The original variable is NOT modified
username=${USER:-"guest"}

# This is useful for configuration with fallback values
config_file=${CONFIG_PATH:-"/etc/default.conf"}

echo "Running as: $username with config: $config_file"

# Related forms:
# ${var-default}   - Only substitute if var is unset (not if empty)
# ${var:=default}  - Same as :- but also ASSIGNS default to var
# ${var:?error}    - Exit with error message if var is unset/empty
# ${var:+alternate} - Use alternate ONLY if var is set (opposite of :-)
```

---

## String Prefix/Suffix Removal

### Use Case: Extract filename without extension or path

```bash
filepath="/home/user/documents/report.pdf"
filename="${filepath##*/}"
name_only="${filename%.*}"
extension="${filename##*.}"
echo "File: $name_only, Extension: $extension"
```

```bash
filepath="/home/user/documents/report.pdf"

# ${var##*/} - Remove longest match of "*/" from the beginning
# ## means "greedy removal from start" - removes "/home/user/documents/"
# Result: "report.pdf"
filename="${filepath##*/}"

# ${var%.*} - Remove shortest match of ".*" from the end
# % means "remove from end" (single % is non-greedy)
# Removes ".pdf", leaving "report"
name_only="${filename%.*}"

# ${var##*.} - Remove longest match of "*." from beginning
# Gets everything after the last dot: "pdf"
extension="${filename##*.}"

echo "File: $name_only, Extension: $extension"

# Pattern memory aid:
# # removes from start (one keyboard position from $)
# % removes from end (shift-4, like $ is shift-4 backwards)
# Single (#/%) is non-greedy (shortest match)
# Double (##/%%) is greedy (longest match)
```

---

## String Replacement

### Use Case: Replace text in strings (like search & replace)

```bash
text="The quick brown fox jumps over the lazy dog"
first_replace="${text/o/O}"
all_replace="${text//o/O}"
echo "First: $first_replace"
echo "All: $all_replace"
```

```bash
text="The quick brown fox jumps over the lazy dog"

# ${var/pattern/replacement} - Replace FIRST occurrence
# Single slash = replace once
first_replace="${text/o/O}"
# Result: "The quick brOwn fox jumps over the lazy dog"

# ${var//pattern/replacement} - Replace ALL occurrences
# Double slash = replace all (global)
all_replace="${text//o/O}"
# Result: "The quick brOwn fOx jumps Over the lazy dOg"

echo "First: $first_replace"
echo "All: $all_replace"

# Other forms:
# ${var/#pattern/replacement} - Replace if matches at START
# ${var/%pattern/replacement} - Replace if matches at END
```

---

## Substring Extraction

### Use Case: Extract portions of strings by position

```bash
phone="555-123-4567"
area_code="${phone:0:3}"
last_four="${phone: -4}"
middle="${phone:4:3}"
echo "Area: $area_code, Exchange: $middle, Last 4: $last_four"
```

```bash
phone="555-123-4567"

# ${var:offset:length} - Extract substring
# Start at position 0, take 3 characters
area_code="${phone:0:3}"  # "555"

# ${var: -n} - Extract last n characters
# Note the SPACE before the minus! Required to distinguish from ${var:-default}
last_four="${phone: -4}"  # "4567"

# Start at position 4, take 3 characters
middle="${phone:4:3}"  # "123"

echo "Area: $area_code, Exchange: $middle, Last 4: $last_four"

# Positions are 0-indexed
# Negative offsets count from end (must have space before -)
# Omitting length means "to the end": ${var:5} gets from position 5 onward
```

---

## String Length

### Use Case: Check or validate string sizes

```bash
password="secret123"
length="${#password}"
if (( length < 8 )); then
    echo "Password too short: $length characters"
fi
```

```bash
password="secret123"

# ${#var} - Get length of string
# The # inside ${} means length (not removal like ## outside)
length="${#password}"  # 9

# (( )) is arithmetic evaluation - compares numbers
if (( length < 8 )); then
    echo "Password too short: $length characters"
fi

# Also works with arrays: ${#array[@]} gives array length
```

---

## Arithmetic Expansion

### Use Case: Perform math calculations

```bash
price=100
tax_rate=8
total=$((price + (price * tax_rate / 100)))
count=5
count=$((count + 1))
echo "Total: \$${total}, Count: $count"
```

```bash
price=100
tax_rate=8

# $(( expression )) - Evaluate arithmetic
# Variables don't need $ inside (( ))
# Supports: + - * / % (modulo) ** (power)
total=$((price + (price * tax_rate / 100)))  # 108

count=5
# Can reference and assign in same operation
count=$((count + 1))  # 6
# Shorthand: ((count++)) or ((count+=1)) also work

echo "Total: \$${total}, Count: $count"

# Note: Bash only does INTEGER math
# For decimals, use bc or awk
# Example: echo "scale=2; 10/3" | bc
```

---

## Brace Expansion

### Use Case: Generate sequences or multiple arguments

```bash
mkdir -p project/{src,test,docs}/{js,css}
touch file{1..5}.txt
echo {A..Z}
mv report.{txt,backup}
```

```bash
# {item1,item2,item3} - Expands to multiple arguments
# Creates: project/src/js, project/src/css, project/test/js, etc.
mkdir -p project/{src,test,docs}/{js,css}

# {start..end} - Numeric or alphabetic sequences
# Creates: file1.txt file2.txt file3.txt file4.txt file5.txt
touch file{1..5}.txt

# Alphabetic ranges also work
echo {A..Z}  # A B C D ... Z

# Common pattern: rename/copy with different extension
# Expands to: mv report.txt report.backup
mv report.{txt,backup}

# Brace expansion happens BEFORE variable expansion
# {1..$var} won't work - use seq or eval instead
# Correct way: touch $(seq 1 $var | xargs -I {} echo "file{}.txt")
```

---

## Array Expansion

### Use Case: Working with arrays of values

```bash
files=("readme.txt" "config.ini" "data.csv")
echo "First: ${files[0]}"
echo "All: ${files[@]}"
echo "Count: ${#files[@]}"
for file in "${files[@]}"; do
    echo "Processing: $file"
done
```

```bash
# Array declaration
files=("readme.txt" "config.ini" "data.csv")

# ${array[index]} - Access specific element (0-indexed)
echo "First: ${files[0]}"  # "readme.txt"

# ${array[@]} - Expands to all elements as separate words
# ALWAYS quote this: "${array[@]}" preserves elements with spaces
echo "All: ${files[@]}"

# ${#array[@]} - Get number of elements
echo "Count: ${#files[@]}"  # 3

# Iterate over array - quotes are critical for spaces in elements
for file in "${files[@]}"; do
    echo "Processing: $file"
done

# ${array[*]} vs ${array[@]}:
# [@] treats each element as separate word (usually what you want)
# [*] joins all elements into single word (rarely useful unless quoted specially)
```

---

## Process Substitution

### Use Case: Use command output as a file

```bash
diff <(ls dir1) <(ls dir2)
while read -r line; do
    echo "Line: $line"
done < <(grep "ERROR" logfile.txt)
```

```bash
# <(command) - Creates a temporary file descriptor with command output
# Useful for commands that need file arguments but you have command output
# Compare directory listings side-by-side
diff <(ls dir1) <(ls dir2)

# >(command) - Creates a file descriptor that pipes to command
# Example: tee to multiple processes
# echo "test" | tee >(process1) >(process2)

# Common pattern: read from command output
# < <(...) means redirect from process substitution
while read -r line; do
    echo "Line: $line"
done < <(grep "ERROR" logfile.txt)

# Why not just pipe? This preserves variable scope:
# while read x; do count=$((count+1)); done < <(cmd)
# vs: cmd | while read x; do count=$((count+1)); done  # count not visible after!
# The pipe version runs in a subshell, losing variable changes
```

---

## Tilde Expansion

### Use Case: Reference home directories

```bash
cd ~/projects
cp file.txt ~alice/shared/
backup_dir=~/.local/backup
```

```bash
# ~ - Expands to current user's home directory
cd ~/projects  # /home/username/projects

# ~user - Expands to specified user's home directory
cp file.txt ~alice/shared/  # /home/alice/shared/

# Common pattern: hidden config directories
backup_dir=~/.local/backup  # /home/username/.local/backup

# Note: Tilde expansion only works at start of word
# $HOME/path works anywhere, ~/path only at beginning
# Won't work: prefix~suffix, "$var~/path"
```

---

## Parameter Expansion - Case Modification

### Use Case: Convert string case

```bash
name="john doe"
upper="${name^^}"
lower="${name,,}"
capitalize="${name^}"
echo "Upper: $upper, Lower: $lower, Capitalized: $capitalize"
```

```bash
name="john doe"

# ${var^^} - Convert all to uppercase
upper="${name^^}"  # "JOHN DOE"

# ${var,,} - Convert all to lowercase
lower="${name,,}"  # "john doe"

# ${var^} - Capitalize first character
capitalize="${name^}"  # "John doe"

# ${var^^pattern} - Convert matching chars to uppercase
# ${var,,pattern} - Convert matching chars to lowercase

echo "Upper: $upper, Lower: $lower, Capitalized: $capitalize"

# Bash 4.0+ feature - won't work in older bash
# Alternative for portability: tr, awk, or typeset -u/-l
```

---

## Indirect Expansion

### Use Case: Variable variables (reference by name)

```bash
env="production"
production_db="prod-server-01"
staging_db="stage-server-01"
database_var="${env}_db"
database="${!database_var}"
echo "Using database: $database"
```

```bash
env="production"
production_db="prod-server-01"
staging_db="stage-server-01"

# Build variable name dynamically
database_var="${env}_db"  # "production_db"

# ${!var} - Indirect expansion
# Gets the value of the variable whose name is in var
# So ${!database_var} looks up $production_db
database="${!database_var}"  # "prod-server-01"

echo "Using database: $database"

# Useful for dynamic configuration
# Alternative pattern: associative arrays (bash 4+)
# declare -A dbs=([production]="prod-01" [staging]="stage-01")
# database="${dbs[$env]}"
```

---

## Quick Reference Summary

| Syntax | Purpose | Example | Result |
|--------|---------|---------|--------|
| `$(cmd)` | Command output | `$(date)` | Command output |
| `!!` | Previous command | `!!` | Re-runs last cmd |
| `!$` | Last arg of prev cmd | `!$` | Last argument |
| `!*` | All args of prev cmd | `!*` | All arguments |
| `!:n` | Nth arg of prev cmd | `!:2` | 2nd argument |
| `${var:-default}` | Default if unset | `${UNSET:-foo}` | "foo" |
| `${var:=default}` | Assign default | Sets var to default | |
| `${#var}` | String length | `${#"hello"}` | 5 |
| `${var:pos:len}` | Substring | `${"hello":1:3}` | "ell" |
| `${var#pattern}` | Remove prefix (short) | `${"file.txt.bak"#*.}` | "txt.bak" |
| `${var##pattern}` | Remove prefix (long) | `${"file.txt.bak"##*.}` | "bak" |
| `${var%pattern}` | Remove suffix (short) | `${"file.txt.bak"%.*}` | "file.txt" |
| `${var%%pattern}` | Remove suffix (long) | `${"file.txt.bak"%%.*}` | "file" |
| `${var/pat/rep}` | Replace first | `${"hello"/l/L}` | "heLlo" |
| `${var//pat/rep}` | Replace all | `${"hello"//l/L}` | "heLLo" |
| `${var^^}` | Uppercase | `${"hello"^^}` | "HELLO" |
| `${var,,}` | Lowercase | `${"HELLO",,}` | "hello" |
| `$((expr))` | Arithmetic | `$((5 + 3))` | 8 |
| `{a,b,c}` | Brace expansion | `file.{txt,md}` | file.txt file.md |
| `{1..5}` | Sequence | `{1..5}` | 1 2 3 4 5 |
| `${arr[@]}` | All array elements | `"${files[@]}"` | Separate words |
| `${#arr[@]}` | Array length | `${#files[@]}` | Count |
| `<(cmd)` | Process substitution | `<(ls)` | Temp file descriptor |
| `~` | Home directory | `~/docs` | /home/user/docs |
| `${!var}` | Indirect expansion | `${!name}` | Value of $name's value |

---

## Common Gotchas

### 1. Quote your variables!
```bash
# WRONG - breaks with spaces
file=$filename
rm $file

# RIGHT - preserves spaces
file="$filename"
rm "$file"
```

### 2. Space before minus in negative substring offset
```bash
# WRONG - interpreted as default value
echo "${var:-3}"

# RIGHT - extracts last 3 chars
echo "${var: -3}"
```

### 3. Arrays must be quoted properly
```bash
# WRONG - breaks elements with spaces
for file in ${files[@]}; do

# RIGHT - preserves each element
for file in "${files[@]}"; do
```

### 4. Arithmetic expansion doesn't need $
```bash
# WORKS but redundant
result=$(($num1 + $num2))

# CLEANER
result=$((num1 + num2))
```

### 5. Brace expansion happens before variable expansion
```bash
# WRONG - doesn't work
end=5
echo {1..$end}  # Outputs: {1..5} literally

# RIGHT - use seq or eval
seq 1 "$end"
# Or: eval "echo {1..$end}"
```

### 6. History expansion re-runs commands
```bash
# DANGER - $(!!) executes the command again
rm file.txt  # Deletes file
result=$(!!)  # Tries to delete again! (will error)

# SAFE - capture output on first run
result=$(rm file.txt)  # Captures stderr/stdout on first run
```
