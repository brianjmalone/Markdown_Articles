# MATLAB to Python: A Survival Guide to Common Traps

This isn't a tutorial. It's a list of traps—the specific mistakes and conceptual hurdles that MATLAB users hit hardest when moving to Python. 

I could add numbers next to each one with how many times I've made each mistake, but it would be embarrassing. 

For instance, using ()'s instead of []'s (Trap #3!) Dropping ()s and returning objects. Adding ()s and getting "not callable" errors. If you've read this far, you get it. 

I made this to review myself, but want to share it in case it helps others. 

Some of these are simple syntax swaps. Others are deeper conceptual shifts that take time to internalize. But all of them are things that will annoy you, so let's get them out there. 

Here's the key to that shift. Python will seem...*wrong*. It's not wrong. It's just different. Google *why* people use zero-indexing, or design documents to try and be more forgiving of the differences and the papercuts of syntactical frictions. 

(It could be worse. You could be reading man pages...)

## Trap 0: The Setup Will Make You Want to Quit

Let's start with the thing that hits you first. This isn't a syntax trap—it's a cultural shift. MATLAB is a product. Python is an ecosystem. Accept that you'll spend time on setup, and know that it gets easier.

**In MATLAB:** You double-click the application. It opens. You start typing code in the editor. You run it. Done.

**In Python:** You need to choose an editor (VS Code? PyCharm? Spyder?). You need to understand virtual environments (`venv`, `conda`). You need to install packages with `pip` or `conda`. You need to figure out which Python version you're using. You need to understand the difference between running a script and running a module.

**The Trap:** You haven't written a line of code yet, and you're already drowning in tooling decisions. A MATLAB user is accustomed to a single, monolithic IDE where everything just works. Python's decentralized ecosystem is powerful and flexible, but it's also overwhelming at first.

**My advice:** Start with Anaconda (it bundles Python, common libraries, and an environment manager). 

Spyder is the most MATLAB-like. Don't give in. Bite the bullet. Work like the people you're likely to be working with work. Open a Terminal window, and run ipython for quick syntax checks as you build. 

 But Jupyter notebooks, YES. Build in a comfortable, easy to iterate place, then refactor for production. LLMs can and will help, there. 
 
 You will have to learn about virtual environments and pip, at least until you can "uv pip" pip away. Embrace the suck. 

 (It sucks.)

But once you get it, even a little, start containerizing. It's nice. venv sprawl is real. There is hope.

There are even *swarms* —eventually. 

## Trap 1: Indexing Starts at Zero (And You'll Forget This Constantly)

Get a screaming pillow with a pleasant texture for this one. 

**In MATLAB:** The first element of any array is at index `1`.

**In Python:** The first element of any list, array, or tuple is at index `0`.

**The Trap:** Off-by-one errors are almost guaranteed, early on. You'll write `my_array[1]` thinking you're getting the first element, but you're actually getting the *second*.

```matlab
% MATLAB
data = [10, 20, 30, 40];
first = data(1);  % 10
```

```python
# Python
data = [10, 20, 30, 40]
first = data[0]  # 10
second = data[1]  # 20
```

Insidious. Pervasive. You'll get it right 80% of the time, then 82%.

Number go up. Keep at it. I hope you don't have to switch back and forth. Oof. 100%? Zeno's Arrow comes to mind. 

**The fix:** There is no fix. This is muscle memory you have to overwrite through repetition. Just know it's coming.

## Trap 2: Slicing Is Exclusive (And So Is `range()`)

**In MATLAB:** Slicing is inclusive on both ends. `1:3` gives you elements 1, 2, and 3. The `end` keyword refers to the last element.

**In Python:** Slicing is exclusive of the upper bound. `0:3` gives you elements 0, 1, and 2 (not 3). The last element is accessed with index `-1`.

**The Trap:** You constantly grab one too many or one too few elements. And the `range()` function follows the same exclusive rule, which compounds the confusion.

```matlab
% MATLAB
data = [10, 20, 30, 40, 50];
subset = data(2:4);  % [20, 30, 40] - three elements
last_three = data(end-2:end);  % [30, 40, 50]
```

```python
# Python
data = [10, 20, 30, 40, 50]
subset = data[1:4]  # [20, 30, 40] - three elements (indices 1, 2, 3)
last_three = data[-3:]  # [30, 40, 50]

# And range() is also exclusive
for i in range(1, 4):  # loops with i = 1, 2, 3 (NOT 4)
    print(i)
```

**Why Python does this:** The exclusive upper bound means `len(data[0:n])` is always `n`, which is mathematically cleaner. And `data[a:b]` followed by `data[b:c]` splits the array perfectly with no overlap or gap. It's elegant once you internalize it, but it feels wrong for months.

## Trap 3: Parentheses vs. Square Brackets (A Syntax Error You'll Make 1000 Times)

**In MATLAB:** Parentheses `()` do everything: calling functions AND indexing arrays.

```matlab
result = my_func(x)  % function call
element = my_array(1)  % indexing
```

**In Python:** Parentheses `()` are ONLY for calling functions/classes. Square brackets `[]` are ONLY for indexing/slicing.

```python
result = my_func(x)  # function call
element = my_array[0]  # indexing
```

**The Trap:** Your fingers will constantly betray you. You'll type `my_list(0)` and get:

```
TypeError: 'list' object is not callable
```

This error message is confusing at first—you're not trying to "call" the list, you're trying to index it! But Python sees `my_list(0)` and thinks you're trying to call `my_list` as a function.

**The fix:** When you see "object is not callable," your first thought should be "did I use parentheses instead of brackets?"

## Trap 4: The Matrix Multiplication You Expect Isn't What You Get

**In MATLAB:** Everything is a matrix. The `*` operator performs matrix multiplication.

```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];
C = A * B;  % Matrix multiplication
```

**In Python (NumPy):** Arrays exist, but `*` does element-wise multiplication. Matrix multiplication uses `@` or `np.dot()`.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A * B  # Element-wise: [[5, 12], [21, 32]]
C = A @ B  # Matrix multiply: [[19, 22], [43, 50]]
C = np.dot(A, B)  # Also matrix multiply: [[19, 22], [43, 50]]
```

**The Trap:** This causes silent logical errors. Your code runs without crashing, but your results are wrong. You multiply two arrays thinking you're doing linear algebra, but you're actually doing element-wise operations.

**Why Python does this:** NumPy is designed for general numerical computing, not just linear algebra. Element-wise operations are far more common in most domains, so `*` does the more frequent operation. Matrix multiplication gets the `@` operator (added in Python 3.5).

This is probably the most dangerous trap on this list because it fails silently.

## Trap 5: Arrays Are Views, Not Copies (Silent Data Corruption Ahead)

This is one of the biggest sources of bugs when moving from MATLAB to Python.

**In MATLAB:** Assignment creates a copy by default.

```matlab
A = [1, 2, 3, 4];
B = A;  % B is a COPY
B(1) = 99;
% A is still [1, 2, 3, 4]
% B is [99, 2, 3, 4]
```

(I said Python is different, not wrong, earlier. Here's where I wonder, though...)

**In Python (NumPy):** Assignment creates a reference, not a copy. Slicing creates a *view*, not a copy.

```python
import numpy as np

A = np.array([1, 2, 3, 4])
B = A  # B is a REFERENCE to A, not a copy
B[0] = 99
# A is now [99, 2, 3, 4] - MODIFIED!
# B is [99, 2, 3, 4]

# Same with slices
C = A[1:3]  # C is a VIEW of A
C[0] = -1
# A is now [99, -1, 3, 4] - MODIFIED AGAIN!
```

**The Trap:** You modify what you think is a separate array, but you're actually modifying the original. This causes data corruption bugs that are incredibly hard to track down because the modification happens far from where you notice the problem.

**The fix:** Use `.copy()` explicitly when you need an independent array:

```python
B = A.copy()  # Now B is truly independent
C = A[1:3].copy()  # Now C is independent
```

**Why Python does this:** Memory efficiency. In scientific computing, arrays can be huge. Creating copies everywhere would be expensive. Python gives you control: views by default, copies when you ask.

## Trap 6: Indentation Isn't Optional, It's Syntax

HEAD. 

TO. 

DESK. 

If there's such a thing as a favorite mistake, this is mine. I enjoy the variety of error messages you get from messing this up!

**In MATLAB:** Code blocks are delimited by the `end` keyword. Indentation is just for readability.

```matlab
for i = 1:10
    if i > 5
        disp(i)
    end
end
```

**In Python:** Code blocks are defined by indentation. There is no `end` keyword. The colon `:` starts a block.

```python
for i in range(1, 11):
    if i > 5:
        print(i)
```

**The Trap:** If your indentation is inconsistent, your code won't run. Mixing tabs and spaces will cause cryptic errors. Forgetting the colon at the end of `if`, `for`, `def`, or `class` lines is a constant mistake.

**Why Python does this:** Readability. Python's creator (Guido van Rossum) believed that if indentation reflects structure anyway, why not make it *be* the structure? It forces you to write readable code.

This feels bizarre at first, especially coming from a language where `end` explicitly closes blocks. But most people grow to like it—no more wondering "which `end` closes which block?"

(Emphasis on "most" there.)

## Trap 7: Functions Don't Return Unless You Tell Them To

Did you get "None" again? 

**In MATLAB:** Return values are declared in the function signature.

```matlab
function y = square(x)
    y = x^2;
end
```

**In Python:** You must explicitly `return` a value. If you don't, the function returns `None`.

```python
def square(x):
    result = x**2
    # Forgot to return!

value = square(5)  # value is None, not 25
```

**The Trap:** You write a function, forget the `return` statement, and your function silently returns `None`. This causes `TypeError` exceptions later when you try to use the result.

**The fix:** If your function should produce a value, it needs a `return` statement. Get in the habit of writing the `return` line first.

It's also common to return True and False when you really mean "This worked!" or "This didn't work, alas..." That can take some getting used to when using methods which change state internally on objects. They seem like *so much* like functions, but differ in important ways, this being one. 

None agrees with my idea of replacing "def" with "meth" in the signature, so, here we are. 

## Trap 8: Scripts Don't Share a Workspace

**In MATLAB:** When you run a script, all variables go into the shared workspace. Run another script, and it can see those variables.

**In Python:** Each `.py` file is a module with its own namespace. Variables defined in one script aren't automatically available in another.

**The Trap:** You run one script, then another, and expect variables to carry over. They don't. You'll also be confused by the `if __name__ == "__main__":` idiom that you see everywhere.

```python
# script_one.py
x = 10
print("Script one ran")

# script_two.py
print(x)  # NameError: name 'x' is not defined
```

To share data between scripts, you need to either:
- Import one script as a module: `from script_one import x`
- Pass data explicitly through function calls
- Use external files or databases

**The `if __name__ == "__main__":` thing:**

Buy a small mirror and have that tattooed in reverse on your forehead. A mini-trap for me is mistyping that particular line of code *every possible way*. 

```python
# my_module.py
def useful_function():
    return 42

if __name__ == "__main__":
    # This code only runs if you execute this file directly
    # It does NOT run if someone imports this file
    print("Testing the function:")
    print(useful_function())
```

This prevents code from running when someone imports your module. In MATLAB, scripts always execute fully. In Python, you need to protect "script-like" code behind this guard.

## Trap 9: Python Has Better Data Structures, But You Have to Use Them

**In MATLAB:** Your tools are matrices, cell arrays, and structs. That's mostly it. Gentle Reader, I love(d) them. 

**In Python:** You have lists, dictionaries, tuples, sets, and more. They're all optimized for different use cases.

**The Trap:** MATLAB users try to force everything into NumPy arrays (the closest thing to MATLAB matrices) even when a different data structure would be far better.

The Hottest Take: I *hate(d?)* Python dictionaries. You know who *loves* them?

**EVERYONE ELSE.** (Certainly everyone who has ever designed a technical interview question...). 

SO. 

MANY. 

KeyErrors. 

A **big** adjustment is finding out you don't have to think in numerical indices all the time. 

```python
for thing in list_of_things...
```

I'm FREE! But range(len()) is always there for you. And have you met my friend enumerate()? 

**Example: When NOT to use an array**

```python
# Bad (MATLAB-style thinking)
data = np.array([["Alice", 25], ["Bob", 30], ["Charlie", 35]])
# Now you have to remember that column 0 is names, column 1 is ages
# And everything is stored as strings because NumPy arrays are homogeneous

# Good (Pythonic thinking)
data = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 35}
]
# Self-documenting, can mix types, easy to access by key
```

Python's built-in data structures are incredibly powerful and efficient. Don't reach for NumPy arrays unless you're actually doing numerical computation.

## Trap 10: Plotting Doesn't Show Up Unless You Ask

Minor, honestly. And if we are being honest, *why are you writing data plotting code yourself?!?!?* 

LLMs are aces at this, so let them do it, then tweak for aesthetics. I simply can't type fast enough to make it worthwhile to do myself, and the number of syntax errors I've seen for plotting code remains rock steady at **ZERO.** Bad design choices? Nonsense labels? Sure, but the frame works. 

**In MATLAB:** Type `plot(x, y)` and a figure window immediately appears.

**In Python (Matplotlib):** Type `plt.plot(x, y)` and... nothing happens. The plot is *staged* but not displayed.

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
plt.plot(x, y)
# Nothing appears yet!

plt.show()  # NOW the plot window appears
```

**The Trap:** You write plotting code in a script, run it, and stare at the screen waiting for the plot to appear. It never does, and you think Matplotlib is broken.

**Why Python does this:** Matplotlib separates "building" the plot from "displaying" it. This gives you more control—you can add multiple elements to a figure before showing it. But it's confusing at first.

**Exception:** In Jupyter notebooks, plots typically appear automatically without `plt.show()` because the notebook has special integration with Matplotlib. That's nice. That's where EDA happens anyway. 

## Trap 11: For-Loops Aren't Evil in Python

**In MATLAB:** For-loops are slow for numerical operations. The culture is to "vectorize or die."

```matlab
% Slow
result = zeros(1, 1000000);
for i = 1:1000000
    result(i) = data(i)^2;
end

% Fast
result = data.^2;
```

**In Python:** For-loops are fast and idiomatic for general-purpose iteration. NumPy vectorization is still critical for *numerical* performance, but Pythonic loops are efficient for many tasks.

**The Trap:** You write complex, unreadable vectorized code to avoid a loop that would have been perfectly fast and much clearer.

```python
# If you're iterating over files, objects, or doing non-numerical work:
for filename in file_list:
    process_file(filename)  # This is fine! No need to vectorize.

# If you're doing numerical computation on arrays:
result = data**2  # Still vectorize this with NumPy
```

Python's loops are implemented in C and are much faster than MATLAB's interpreted loops. Don't avoid them reflexively.

**BE FREE!**

## Trap 12: String Manipulation Is Actually Pleasant

**In MATLAB:** Strings were originally character arrays, making manipulation awkward. The newer `string` type helps, but it's not central to the language.

It's better in newer versions, but it was bad for a while. 

**In Python:** Strings are a powerful, fundamental data type with an enormous library of built-in methods.

**The Trap:** You write manual loops to parse or manipulate strings, not realizing Python has a one-liner for it.

```python
text = "  hello world  "

text.strip()  # "hello world" - removes whitespace
text.upper()  # "  HELLO WORLD  "
text.replace("world", "Python")  # "  hello Python  "
text.split()  # ["hello", "world"] - splits on whitespace

# Joining strings
words = ["hello", "world"]
" ".join(words)  # "hello world"
"-".join(words)  # "hello-world"

# Checking contents
text.startswith("hello")  # False (leading spaces)
text.strip().startswith("hello")  # True
"world" in text  # True
```

Python's string methods are fast, readable, and comprehensive. Learn them. They'll save you hours of manual parsing.

Have a string parsing issue? Ask your favorite LLM *first*, and learn a new string method to love. 

## Trap 13: Comments Use `#`, Not `%`

This is trivial, but you'll mess it up constantly for the first few weeks. 

(And then, months later, again, just once, for no apparent reason.) 

**MATLAB:**
```matlab
% This is a comment
```

**Python:**
```python
# This is a comment
```

Your muscle memory will betray you. You'll type `%` and wonder why your code won't run. It's a simple fix, but it's annoying.

## Trap 14: The Standard Library Isn't Always Obvious

**In MATLAB:** You have official Toolboxes from MathWorks. Image Processing Toolbox. Statistics Toolbox. Signal Processing Toolbox. It's curated and (mostly) consistent.

**In Python:** You have thousands of open-source libraries. They're free, powerful, and community-developed. But there are often multiple competing options for the same task.

**The Trap:** You don't know which library is "standard" for a given task. And version conflicts between libraries can be painful.

**The core stack for scientific Python:**
- **NumPy**: Arrays and numerical computation
- **Pandas**: Tabular data (think Excel or MATLAB tables)
- **Matplotlib**: Plotting (MATLAB-style interface)
- **SciPy**: Scientific algorithms (optimization, integration, signal processing)
- **scikit-learn**: Machine learning
- **scikit-image**: Image processing

These are the "blessed" libraries. Start here before exploring alternatives.

## Trap 15: Broadcasting Rules Are Slightly Different

Both MATLAB and NumPy support broadcasting—automatically expanding arrays of different shapes to make operations work.

**The Trap:** NumPy's broadcasting rules can be stricter or more permissive in subtle ways, leading to unexpected `ValueError` exceptions about incompatible shapes.

```python
import numpy as np

A = np.array([[1, 2, 3]])  # Shape: (1, 3)
B = np.array([[1], [2], [3]])  # Shape: (3, 1)

C = A + B  # Shape: (3, 3) - broadcasts correctly
# Result:
# [[2, 3, 4],
#  [3, 4, 5],
#  [4, 5, 6]]

# But sometimes you get:
# ValueError: operands could not be broadcast together with shapes (X,) (Y,)
```

**The fix:** When you hit broadcasting errors, explicitly `reshape()` or add dimensions with `np.newaxis` until the shapes are compatible.

This is an advanced trap. You'll encounter it when you start doing more complex array operations.

## The Path Forward

These traps are real, and you'll hit most of them. Some multiple times. 

SO. 

MANY. 

TIMES.

At least I did. 

The transition from MATLAB to Python isn't just learning new syntax—it's learning a new philosophy. MATLAB is a product optimized for matrix math in an integrated environment. Python is an ecosystem optimized for flexibility and general-purpose programming. It does MOAR. Really. 

In fact, people have built so much code AROUND it, that it can be tricky to know when you're looking at "Python" or you're using custom methods from a class from a module from a framework on a turtle on a turtle on another turtle...

**Some advice:**

1. **Start with Jupyter notebooks.** They're interactive like the MATLAB command window, and you can see results immediately.

2. **Don't fight Python's idioms.** For-loops are okay. Dictionaries are powerful. Hateful. Tricksy. FALSE! But powerful. Not everything needs to be an array. (Single tear...)

3. **Learn the standard libraries.** NumPy, Pandas, Matplotlib. NumPy will make you feel right at home. ("Have you met my friend, Jax?")

4. **Read other people's code.** Python has a strong culture of readable, "Pythonic" code. The more you read, the faster you'll internalize the patterns.

Honestly, it's incredibly demystifying to do this. Go on GitHub and find code from organizations you admire, so you can admire the code they write. Have an LLM explain to you the parts you don't get. Get a feel for how deep the rabbit hole goes early on. 

5. **Accept that it takes time.** You'll be slower in Python than MATLAB for a few months. That's okay. The flexibility and ecosystem are worth it.

(But you know who's faster and more accurate in Python? ALL of the frontier models. It's a huge compensatory advantage of the Python ecosystem.)

When you find yourself frustrated when you have to go *back* to MATLAB, you've won! 

Now go write some Python, debug, and come back here when you hit one of these.

## A Trap-Filled Example: What MATLAB Users Write on Day One

Here's what MATLAB-transplant code looks like. I've marked each trap with its number. See how many you can spot before looking back:

```python
# Let's analyze some data!

import numpy as np
import matplotlib.pyplot as plt

def process_data(data, threshold)  # Trap 6
    n = len(data)
    results = np.zeros(n)

    # Loop through and square values above threshold
    for i in range(1, n):  # Traps 1 & 2
        if data(i) > threshold  # Traps 3 & 6
            results(i) = data(i) * data(i)  # Trap 3 (x3)
        end  # Trap 6
    end  # Trap 6

    # Get elements 1 through 5
    subset = results(1:5)  # Traps 2 & 3

    # Calculate mean but forget to return it
    mean_val = np.mean(subset)  # Trap 7

# Process my data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
filtered = process_data(data, 5)  # Trap 7

# Make a backup copy
backup = data  # Trap 5
backup(1) = 999  # Traps 1, 3, & 5

# Do some matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A * B  # Trap 4

# Plot the results
plt.plot(data, filtered)  # Trap 10
plt.xlabel('Data Points')
plt.ylabel('Filtered Values')
# Where's my plot?!
```

**Trap count: 10+ distinct traps in 30 lines.** Can you find them all?

This is *exactly* what I wrote, multiple times, in various forms, for months. If you see yourself in this code, congratulations—you're normal. Every MATLAB user writes this. The difference between you and a Python programmer is just time and repetition.

Then, come back here and know that you're not alone! 

This article is a helmet for the next time you go running through this particular field of rakes. Strap it on, and sprint. 

---

## Bonus Round: NumPy-Specific Nightmares

Once you get past the basic traps, NumPy has its own special torture chamber.

### The `[0]` Extraction Tax

**In MATLAB:** Indexing a scalar from an array gives you a scalar.

```matlab
result = mean(data);  % result is a scalar: 5.0
x = result + 10;  % Works fine: 15.0
```

**In Python (NumPy):** Sometimes you get a scalar, sometimes you get a zero-dimensional array, sometimes you get a single-element array. You need to extract it.

```python
import numpy as np

# Sometimes it's fine
result = np.mean(data)  # numpy.float64 - acts like a scalar
x = result + 10  # Works: 15.0

# But other operations return arrays
result = np.array([5.0])  # Single-element array
x = result + 10  # Still an array: array([15.])
x = result[0] + 10  # Now a scalar: 15.0

# Or worse, zero-dimensional arrays
result = np.sum(data, keepdims=True)
# Now you need to do result.item() or result[()] to extract
```

**The Trap:** You'll write code that works with some operations but breaks with others. You'll spend hours debugging why `if result > 5:` sometimes works and sometimes gives you "truth value of array is ambiguous."

**The Fix:** When you need a true Python scalar, use `.item()` or `[0]` or `float()` to extract it explicitly. Or just embrace that NumPy scalars exist and mostly work fine.

### The `axis` Parameter Betrayal

**In MATLAB:** Dimensions are numbered 1, 2, 3... Operations default to the first non-singleton dimension.

```matlab
A = [1, 2, 3; 4, 5, 6];
sum(A, 1)  % Sum down the rows (dimension 1): [5, 7, 9]
sum(A, 2)  % Sum across the columns (dimension 2): [6; 15]
```

**In Python (NumPy):** Axes are numbered 0, 1, 2... and `axis=0` means "collapse the first dimension" which is DOWN the rows (the opposite intuition).

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
np.sum(A, axis=0)  # [5, 7, 9] - sums down rows (collapses axis 0)
np.sum(A, axis=1)  # [6, 15] - sums across columns (collapses axis 1)
```

**The Trap:** You'll confuse "axis number" with "dimension you're operating along." In MATLAB, dimension 1 is rows. In NumPy, axis=0 is also rows, but it COLLAPSES that axis. Same result, different mental model.

**The Fix:** Think of `axis` as "which dimension to remove" not "which dimension to operate on."

### Boolean Indexing Looks Similar But Isn't

**In MATLAB:**
```matlab
data = [1, 2, 3, 4, 5];
mask = data > 2;  % [false, false, true, true, true]
result = data(mask);  % [3, 4, 5]
```

**In Python (NumPy):**
```python
data = np.array([1, 2, 3, 4, 5])
mask = data > 2  # array([False, False, True, True, True])
result = data[mask]  # array([3, 4, 5])
```

**Wait, that looks the same!** It mostly is, but:

**The Trap:** In MATLAB, you can use `find()` to get indices, then index with those. In NumPy, `np.where()` returns a tuple that needs unpacking.

Gentle Reader, you **are** going to forget to unpack it. 

```matlab
% MATLAB
indices = find(data > 2);  % [3, 4, 5]
result = data(indices);  % Works
```

```python
# Python
indices = np.where(data > 2)  # Tuple: (array([2, 3, 4]),)
result = data[indices]  # Works, but indices is a tuple
result = data[indices[0]]  # More explicit

# Or just use boolean indexing
result = data[data > 2]  # Cleaner
```

### Row-Major vs Column-Major (Advanced Trap)

**In MATLAB:** Arrays are stored in column-major order (Fortran-style). This affects performance when you iterate.

**In Python (NumPy):** Arrays are stored in row-major order (C-style) by default.

**The Trap:** If you're doing heavy numerical work and caring about memory layout, you'll need to think about this. For most people, it's invisible until it causes a mysterious 10x slowdown.

**The Fix:** For beginners, ignore this. For advanced users working with large arrays and interfacing with C/Fortran libraries, read about `np.ascontiguousarray()` and `np.asfortranarray()`.

### The Shape Gotcha: `(n,)` vs `(n,1)`

**In MATLAB:** There's no distinction between a row vector `[1, 2, 3]` and a column vector `[1; 2; 3]` for most operations. They're both 1D.

**In NumPy:** These are different shapes, and it matters.

```python
row = np.array([1, 2, 3])  # Shape: (3,) - 1D array
col = np.array([[1], [2], [3]])  # Shape: (3, 1) - 2D array

# They behave differently in operations
row + row  # Works: array([2, 4, 6])
col + col  # Works: array([[2], [4], [6]])
row + col  # Broadcasting! array([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
```

**The Trap:** You'll create a 1D array when you meant to create a 2D column vector, then matrix multiplication fails or broadcasting does something unexpected.

**The Fix:** Use `reshape()` or `[:, np.newaxis]` to add dimensions when needed:

```python
row = np.array([1, 2, 3])  # Shape: (3,)
col = row[:, np.newaxis]  # Shape: (3, 1)
```

---

## Converting Existing MATLAB Code: A Survival Strategy

So you have 10,000 lines of MATLAB code and need to port it to Python. I'm sorry. Sucks to be you. 

(I say that as someone it has sucked to be for precisely this reason.)

### Rule #1: Flatten Your Data Structures FIRST

**The Problem:** MATLAB loves nested structs. You probably have things like:

```matlab
results.experiment1.data.raw.voltage = [1, 2, 3];
results.experiment1.data.raw.current = [4, 5, 6];
results.experiment1.params.sampling_rate = 1000;
```

When you load this with `scipy.io.loadmat()`, you get a horrific nested array structure that looks like:

```python
data = loadmat('results.mat')
# data is a dict containing arrays containing structs containing arrays containing...
# You'll need something like:
voltage = data['results'][0,0]['experiment1'][0,0]['data'][0,0]['raw'][0,0]['voltage'][0]
```

**See all those `[0,0]` indices?** That's because MATLAB structs become arrays of structured arrays. It's a nightmare.

**The Fix:** Flatten your data structures in MATLAB BEFORE saving, or immediately after loading in Python.

```python
# After loading the nested horror
voltage = data['results'][0,0]['experiment1'][0,0]['data'][0,0]['raw'][0,0]['voltage'][0]

# Immediately flatten into a simple dictionary
flat_data = {
    'voltage': voltage,
    'current': current,
    'sampling_rate': sampling_rate
}

# Or even better, use pandas
import pandas as pd
df = pd.DataFrame({
    'voltage': voltage.flatten(),
    'current': current.flatten()
})
```

**Better yet:** If you still have access to MATLAB, save to a simpler format:

```matlab
% In MATLAB, flatten before saving
save('results.csv', 'voltage', 'current', '-ascii')
% Or use parquet, HDF5, or JSON for structured data
```

Then load it in Python with pandas:

```python
df = pd.read_csv('results.csv')
```

### Rule #2: MATLAB Tables → Pandas DataFrames

If you're using MATLAB's table data type, rejoice! This maps naturally to pandas.

```matlab
% MATLAB
T = table([1;2;3], [4;5;6], 'VariableNames', {'A', 'B'});
```

```python
# Python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
```

Pandas is more powerful than MATLAB tables. Learn it. Love it. It's worth the investment.

### Rule #3: Use `scipy.io.loadmat()` but Prepare for Pain

```python
from scipy.io import loadmat

data = loadmat('myfile.mat')
# data is a dictionary
# But every MATLAB variable becomes a numpy array, even scalars
# Structs become structured arrays with [0,0] indexing hell

# Immediately inspect what you got
print(data.keys())  # See what variables are in there
print(data['myvar'].shape)  # See the shape
print(data['myvar'].dtype)  # See if it's a struct
```

**Pro tip:** Use `squeeze_me=True` and `struct_as_record=False` to reduce some of the pain:

```python
data = loadmat('myfile.mat', squeeze_me=True, struct_as_record=False)
# This helps, but it's still not perfect
```

### Rule #4: Refactor Incrementally

Don't try to convert everything at once. Here's a workflow:

1. **Start with one function** - pick something pure and numerical
2. **Write tests** - use your MATLAB outputs as test cases
3. **Convert function by function** - keep MATLAB running in parallel to compare outputs
4. **Use `numpy.testing.assert_allclose()`** to verify numerical equivalence
5. **Refactor to be Pythonic after it works** - don't optimize prematurely

```python
import numpy as np
from numpy.testing import assert_allclose

# Your converted function
python_result = my_python_function(data)

# Load MATLAB output for same input
matlab_result = loadmat('matlab_output.mat')['result']

# Verify they match
assert_allclose(python_result, matlab_result, rtol=1e-5)
```

### Rule #5: Don't Convert, Rewrite

Sometimes the best "conversion" is to not convert at all. If your MATLAB code is:
- Old and messy
- Poorly documented
- Full of legacy workarounds
- Using outdated algorithms

Consider: **rewriting from scratch** using modern Python libraries. You'll often end up with cleaner, faster, shorter code.

Example: That 500-line MATLAB image processing pipeline? Might be 50 lines with scikit-image.

---

## Quick Wins: Things That Are Actually Better in Python

After all this pain, here's the payoff. Some things are genuinely nicer in Python:

### List Comprehensions

**MATLAB doesn't have these.** You have to write loops or use cellfun/arrayfun awkwardly.

```python
# Python
squares = [x**2 for x in range(10)]
filtered = [x for x in data if x > 5]
pairs = [(x, y) for x in range(3) for y in range(3)]
```

This is readable, fast, and incredibly useful.

### Generator Expressions (Even Better)

**This is where Python gets magical.** Generator expressions look like list comprehensions but use parentheses. They're *lazy*—they don't build the whole list in memory.

```python
# List comprehension - builds entire list immediately
squares_list = [x**2 for x in range(1000000)]  # Uses lots of memory

# Generator expression - computes values on demand
squares_gen = (x**2 for x in range(1000000))  # Uses almost no memory

# Use it in a loop
for square in squares_gen:
    if square > 1000:
        break  # Stops early, never computed the rest!

# Or pass directly to functions that consume iterables
total = sum(x**2 for x in range(1000000))  # No intermediate list!
max_val = max(x for x in data if x > 0)  # Filters on the fly
```

**Why this is amazing for MATLAB users:**

In MATLAB, you're used to creating entire arrays, even when you'll only use them once:

```matlab
% MATLAB - creates full array in memory
values = 1:1000000;
squares = values.^2;
total = sum(squares);  % Array no longer needed after this
```

In Python with generators:

```python
# Python - never creates the full array
total = sum(x**2 for x in range(1000000))  # Memory efficient!
```

For large datasets, this is a game-changer. You can process gigabytes of data without loading it all into RAM.

### Dictionary Comprehensions

```python
# Python
word_lengths = {word: len(word) for word in ['cat', 'dog', 'elephant']}
# {'cat': 3, 'dog': 3, 'elephant': 8}
```

MATLAB structs are clunky by comparison.

### F-Strings for Formatting

```python
name = "Alice"
age = 30
print(f"{name} is {age} years old")
print(f"In 5 years, {name} will be {age + 5}")
```

So much cleaner than MATLAB's `sprintf` or `fprintf`.

### Context Managers

```python
with open('file.txt', 'r') as f:
    data = f.read()
# File automatically closed, even if an error occurred
```

No more forgetting to `fclose()`.

### The Entire Ecosystem

Need to:
- Parse JSON? `import json` (built-in)
- Make HTTP requests? `import requests` (one line install)
- Build a web dashboard? `import streamlit` or `import dash`
- Do machine learning? `import scikit-learn` or `import torch`
- Process images? `import PIL` or `import cv2`
- Work with dates? `import datetime` or `import pandas`

Python has mature, well-documented libraries for everything. MATLAB's toolboxes are expensive and limited.

### It's Free

You can install Python on a cluster, a raspberry pi, a web server, your phone, anywhere. No license server. No fees. No restrictions. This is huge for reproducibility and collaboration.

---

When you find yourself frustrated when you have to go *back* to MATLAB, you've won! 
