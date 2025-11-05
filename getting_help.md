
# How to Get Unstuck in Python
*A Field Manual for Discovering What You Can Do—and How*

There isn't one best way. Context matters. 

Some approaches are better than others, and those depend on your personality. For example, I am insatiably curious, and I risk "rabbit-holing", "squirrel-chasing", etc. For me, getting an *answer* fast without the beckoning embrace of the Whole Internet has special value in preserving flow. 

The goal is to solve problems with minimal context switching.

---
## Table of Contents
- [The Core Workflow: The `Command-Tab` REPL](#the-core-workflow-the-command-tab-repl)
- [Discovery Tools: Finding the Right Words](#discovery-tools-finding-the-right-words)
- [Inspection Tools: `help()` and `?`](#inspection-tools-help-and-)
- [Applied Workflow: Finding a File](#applied-workflow-finding-a-file)
- [How to Read Function Signatures](#how-to-read-function-signatures)
- [Debugging Your Confusion with Errors](#debugging-your-confusion-with-errors)
- [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## The Core Workflow: The `Command-Tab` REPL

Your most powerful tool is a second window.
1.  Open your editor.
2.  Open a terminal with `ipython` running.
3.  Place them side-by-side.

When you're editing a `.py` file and get stuck, don't go to your browser. `Command-Tab` (or `Alt-Tab`) to your IPython console. Recreate the object you're struggling with and poke at it. Test your assumptions. This loop—edit, swap, test, swap back—is faster than any other method of learning.

---

## Discovery Tools: Finding the Right Words

These tools answer: "What can I do with this thing?"

### 1. `TAB` Completion (The Default)
This is your primary tool. It's fast, inline, and visual.
```python
my_string = "hello"
my_string.<TAB>  # Shows all string methods
```

### 2. `dir()` (The Inventory)
When `TAB` is too noisy or you're in a plain REPL, `dir()` gives you a complete, alphabetized list.
```python
import pandas as pd
dir(pd) # See everything in the pandas namespace
```
**Pro-Tip:** Filter `dir()` output to find what you need faster.
```python
# Find all pandas functions related to 'csv'
[name for name in dir(pd) if 'csv' in name]
# >> ['read_csv', 'to_csv']
```

---

## Inspection Tools: `help()` and `?`

Once you have a name, these tools answer: "How do I use it?"

- **`object?`** (IPython/Jupyter): Clean, concise signature and docstring. Your go-to.
- **`object??`** (IPython/Jupyter): Shows the source code. The ultimate ground truth.
- **`help(object)`** (Everywhere): The verbose, comprehensive view. Use it when `?` isn't enough.

---

## Applied Workflow: Finding a File

Let's tie it together. You want to find a specific `.md` file in the current directory.

**1. Discover the tool.** You know the `os` module handles file stuff.
```python
import os
# What's in here?
[name for name in dir(os) if 'dir' in name]
# >> ['fchdir', 'listdir', 'mkdir', 'makedirs', 'rmdir', 'scandir']
```
`listdir` looks promising.

**2. Inspect the tool.** How does it work?
```python
os.listdir?
# Signature: os.listdir(path=None)
# Docstring: Return a list containing the names of the entries in the directory given by path.
```
Simple enough. It takes a path and returns a list of names.

**3. Execute and refine.**
```python
all_files = os.listdir('.')
# Now, filter that list for the file you want
[f for f in all_files if f.endswith('.md')]
```
This entire process happened without leaving the REPL. That's the goal.

---

## How to Read Function Signatures

Signatures tell you what a function needs. `split(sep=None, maxsplit=-1) -> list[str]` means:
- **`sep=None`**: Optional argument with a default value.
- **`maxsplit=-1`**: Another optional argument.
- **`-> list[str]`**: Returns a list of strings.

**Strategy:** Identify required arguments (the ones with no `=`). Ignore the rest until you need them.

---

## Debugging Your Confusion with Errors

Errors are your friends. They tell you exactly what Python is confused about.

- **`AttributeError: 'str' object has no attribute 'split_string'`**
  - **Meaning:** You called a method that doesn't exist.
  - **Debug:** `dir('hello')` to see the *actual* method names. You'll see `split`, not `split_string`.

- **`TypeError: unsupported operand type(s) for +: 'int' and 'str'`**
  - **Meaning:** You tried to combine two types that don't know how to work together.
  - **Debug:** `type(my_variable)` to check your assumptions. Then find the right way to combine them (e.g., `str(my_number)`).

- **`NameError: name 'my_variable' is not defined`**
  - **Meaning:** You used a variable that hasn't been assigned.
  - **Debug:** Check for typos or run the cell that defines the variable.

---

## Quick Reference Cheat Sheet

| Goal                               | Command                               |
| ---------------------------------- | ------------------------------------- |
| **See what's possible**            | `object.<TAB>`                        |
| **Get a full list of methods**     | `dir(object)`                         |
| **Find a specific method**         | `[m for m in dir(obj) if 'x' in m]`   |
| **Read the short docs**            | `object.method?`                      |
| **Read the full docs**             | `help(object.method)`                 |
| **Read the source code**           | `object.method??`                     |
| **Check an object's type**         | `type(object)`                        |

