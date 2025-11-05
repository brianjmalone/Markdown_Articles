# While Loops: Timing and Control Flow

## When the Condition is Checked

**The while loop condition is evaluated at exactly ONE moment: at the beginning of each iteration, before entering the loop body.**

```python
x = 0
while x < 3:
    print(f"Start: x = {x}")
    x += 1
    print(f"End: x = {x}")
    # Loop doesn't stop here even if condition is now false
    # It always completes the entire iteration first!

# Output:
# Start: x = 0
# End: x = 1
# Start: x = 1
# End: x = 2
# Start: x = 2
# End: x = 3
```

**Key insight:** Even if something inside the loop body makes the condition false, the loop completes the entire iteration before checking again.

---

## Three Common While Loop Patterns

### 1. Condition at Top (Most Common)

```python
while condition:
    # do work
```

**Use when:**
- The condition naturally describes "when to keep going"
- You want to check BEFORE doing any work
- Most readable, idiomatic Python

**Example:**
```python
while index not in visited:
    visited.add(index)
    # process
```

### 2. Break at End (Do-While Pattern)

```python
while True:
    # do work
    if condition:
        break
```

**Use when:**
- You ALWAYS need to run the loop body at least once
- The exit condition depends on work you just did
- You want ALL work to complete before checking to exit

**Example:**
```python
while True:
    user_input = input("Command: ")
    process(user_input)
    if user_input == "quit":
        break
```

### 3. Break at Start (Guard Pattern)

```python
while True:
    if condition:
        break
    # do work
```

**Use when:**
- Exit condition is met but you DON'T want to do the work
- Need early exit before expensive operations
- Exit check is conceptually separate from main loop logic

**Example:**
```python
while True:
    if not has_more_data():
        break
    expensive_operation()
```

---

## Order of Operations Matters

**The sequence inside the loop affects results, not iteration count:**

```python
# Version 1: total = 10
count = 0
total = 0
while count < 5:
    total += count  # Add THEN increment
    count += 1

# Version 2: total = 15  
count = 0
total = 0
while count < 5:
    count += 1      # Increment THEN add
    total += count
```

**Both run 5 iterations. Different order = different results.**

---

## Critical Timing Example

**Problem:** Track visited positions, stop when revisiting.

```python
# WRONG - marks NEW position, not current
while index not in visited:
    index = get_next(index)  # Change index first
    visited.add(index)       # Mark new position

# CORRECT - marks current position before moving
while index not in visited:
    visited.add(index)       # Mark where you ARE
    index = get_next(index)  # Then move
```

**Rule:** Mark/use values BEFORE modifying them if you need the original value.

---

## Quick Reference

| Pattern | When to Use | Example Use Case |
|---------|-------------|------------------|
| `while condition:` | Clean, readable exit condition | Processing until state changes |
| `while True:` + break at end | Must run once, check after work | User input loops |
| `while True:` + break at start | Check before expensive work | Resource availability checks |

**Remember:** The condition check happens at the TOP. Everything else is about controlling what work happens when.