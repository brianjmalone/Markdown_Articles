# Python Gotchas: A Reference Guide

## Reference vs Copy
```python
# Assignment creates reference, not copy
x = [1, 2, 3]
y = x           # y and x point to SAME list
y.append(4)     # x is now [1, 2, 3, 4]

# To actually copy
y = x[:]        # or x.copy() or list(x)
```

## In-Place Methods Return None
```python
x = [3, 1, 2]
result = x.sort()  # result = None
# x is now [1, 2, 3]

# Same for: append(), reverse(), extend()
```

## += Behavior Varies by Type
```python
# Lists: modifies in place
x = [1, 2, 3]
y = x
x += [4]        # y is now [1, 2, 3, 4]

# But + creates new list
x = [1, 2, 3]
y = x
x = x + [4]     # y is still [1, 2, 3]

# Tuples: += creates new object
x = (1, 2, 3)
y = x
x += (4,)       # y is still (1, 2, 3)
```

## Mutable Default Arguments
```python
def foo(x = []):
    x.append(1)
    return x

foo()  # [1]
foo()  # [1, 1]  # Same list persists across calls

# Correct pattern
def foo(x = None):
    if x is None:
        x = []
    x.append(1)
    return x
```

## is vs ==
```python
# Small integers are cached
a = 10
b = 10
a is b  # True

a = 1000
b = 1000
a is b  # False (usually, context-dependent)

# Strings may be interned
a = "hello"
b = "hello"
a is b  # True (sometimes)

# Rule: use == for values, is only for None
if x is None:  # Correct
if x == None:  # Works but not idiomatic
```

## Without terminal '()' you get the method, not the effect 
```python
text = "hello"
print(text.upper)
# <built-in method upper of str object at 0x1089ba6b0>
```

## String Immutability
```python
text = "hello"
text[0] = "H"  # TypeError

# Must create new string
text = "H" + text[1:]
```

## List Operators
```python
[1, 2] + [3, 4]     # [1, 2, 3, 4]
[1, 2] * 3          # [1, 2, 1, 2, 1, 2]
[1] * 3             # [1, 1, 1]

# Caution with mutable objects
[[]] * 3            # [[], [], []] - all same object
x = [[]] * 3
x[0].append(1)      # [[1], [1], [1]]
```

## Slice Assignment
```python
nums = [1, 2, 3, 4, 5]

# Single index: replaces element with object
nums[2] = []        # [1, 2, [], 4, 5]

# Slice: unpacks iterable
nums[2:4] = []      # [1, 2, 5]
nums[2:4] = [7]     # [1, 2, 7, 5]
nums[2:4] = [7,8,9] # [1, 2, 7, 8, 9, 5]
```

## Function Parameters
```python
# Lists passed by reference - modifies inputs without reassignment
def modify(lst):
    lst.append(4)

nums = [1, 2, 3]
modify(nums)
print(nums)  # [1, 2, 3, 4]

# Reassignment doesn't affect original- phantom name-sharing
def modify(lst):
    lst.append(4)    # Modifies original
    lst = [1, 2, 3]  # Only affects local variable

nums = [0]
modify(nums)
print(nums)  # [0, 4]
```

## Slice Notation (the Final Element Tax)
```python
x = [0, 1, 2, 3, 4, 5]
x[1:4]   # [1, 2, 3] - includes start, excludes stop
x[:3]    # [0, 1, 2]
x[3:]    # [3, 4, 5]
x[:]     # [0, 1, 2, 3, 4, 5] - shallow copy
```

## Ternary Operator (You can only return once)
```python
# Syntax: value_if_true if condition else value_if_false
result = 1 if x > y else -1

# Not: return 1 if x > y else return -1
```

## range() Behavior (I want x things—but not x)
```python
range(5)      # 0, 1, 2, 3, 4
range(1, 5)   # 1, 2, 3, 4
range(0, 10, 2)  # 0, 2, 4, 6, 8
```

## Dictionary get() Method (use defaultdict, stay sane)
```python
d = {'a': 1, 'b': 2}
d['c']           # KeyError
d.get('c', 0)    # 0 (returns default)
```

## String Methods (showing without doing)
```python
"hello".upper()      # "HELLO"
"HELLO".lower()      # "hello"
"  text  ".strip()   # "text"
"hello".replace("l", "x")  # "hexxo"

# Strings are immutable - methods return new strings
text = "hello"
text.upper()  # Returns "HELLO" but text unchanged
text = text.upper()  # Must reassign to change text variable
```

## String Padding with initial zeros (argument sets final length)
```python
"56".zfill(4)     # "0056"
"123".zfill(5)    # "00123"
"99999999".zfill(0)   # "99999999"
```

## Multiple Ways to Copy Lists (not so Zen)
```python
x = [1, 2, 3]

# Method 1
y = x[:]

# Method 2
y = x.copy()

# Method 3
y = list(x)

# Method 4
import copy
y = copy.copy(x)
```

## Multiple Ways to Check Equality
```python
x = [1, 2, 3]
y = [1, 2, 3]

x == y   # True (value equality)
x is y   # False (identity check)

# For cached objects (tricky!)
a = 10
b = 10
a == b   # True
a is b   # True (cached)
```

## Multiple Ways to Modify Lists (not so Zen)
```python
x = [1, 2, 3]

# Method 1
x.append(4)      # [1, 2, 3, 4]

# Method 2
x += [4]         # [1, 2, 3, 4]

# Method 3
x.extend([4])    # [1, 2, 3, 4]

# Method 4
x = x + [4]      # Creates new list
```

## Iteration Patterns 
```python
items = ['a', 'b', 'c']

# Index only
for i in range(len(items)):
    print(i, items[i])

# Value only
for item in items:
    print(item)

# Both index and value
for i, item in enumerate(items):
    print(i, item)
```