# Things to Know

This document is a summary of useful commands, tips, and tricks, organized by topic.

## Bash & Shell

*   **Toggle line numbers in nano:** `Escape` + `N`
*   **Count elements in a bash array:** `echo ${#array[@]}`
*   **Count length of the first element in a bash array:** `echo ${#array[0]}`
*   **Create an array from a string:** `IFS=' ' read -ra array <<< 'onestring twostring threestring fourstring'`
*   **Here string syntax (`<<<`):** Pass strings directly to commands that expect standard input.
    *   `wc <<< "How many lines/words/characters are in this sentence?"`
    *   `bc <<< "scale=2; 76543/23456"` (built-in calculator)
*   **`shellcheck`:** A tool for debugging bash scripts (available at shellcheck.net).
*   **Loop over array indices:** `for i in ${!myArray[@]}; do`
*   **Test expressions:** Use `||` for a concise test. `[[ "${#arr[@]}" -eq 1 ]] || echo 'Yup'`
*   **Add to an array:** `arr+=( "x" )`
*   **List array contents:** `for str in ${arr[@]}; do echo $str; done`
*   **Running scripts:**
    *   `./script.sh`: Runs in a child process.
    *   `bash ./script.sh`: Specifies the interpreter, similar to the above.
    *   `source ./script.sh`: Runs in the current shell (necessary for commands like `cd`).
*   **Increment a variable:** `$(( x++ ))`
*   **Test if a variable is set:** `if [ -v $counter ]`
*   **File Descriptors:**
    *   `0`: stdin (keyboard)
    *   `1`: stdout (terminal)
    *   `2`: stderr (terminal)
*   **History commands:**
    *   Show history from line 100 onwards: `history | tail -n +100`
    *   Show history from lines 100-125: `history | tail -n +100 | head -n 25`
    *   Show longest commands: `history | awk '{print length, $0}' | sort -nr | head -n 20`
*   **`exec`:** Replaces the current process. Use `(exec echo "Hey")` to run in a subshell and avoid closing the terminal.
*   **Get current process ID (PID):** `echo $$`
*   **`find` command:**
    *   `-exec` needs a `{}` placeholder for found items and must be terminated with `\;` or `+`.
    *   List files from longest to shortest: `find . -maxdepth 1 -type f -exec wc -l {} + | sort -r`
    *   Find and rename with string replacement: `find . -type f -name "crap*.butts" -exec bash -c 'mv "$1" "${1%.butts}.holes"' _ {} \;`
*   **Return values from Python to bash:** `variable=$(python -c "import my_script; print(my_script.my_function())")`
*   **Test email regex:** `[[ "$email" =~ ^[[:alnum:]_]+@[[:alnum:]_]+\\.[[:alnum:]_]+$ ]]`
*   **Get last exit status:** `echo $?`
*   **Create an array of files:** `filelist=(*.ext)`
*   **Access array elements:**
    *   First file: `${filenames[0]}`
    *   Slice (files 2, 3, 4): `${filenames[@]:2:3}`
    *   First 5 files: `${filenames[@]:0:5}`
*   **Get array indices:** `echo ${!my_array[@]}`
*   **Clear a variable:** `unset varname`
*   **`fc` command:**
    *   `fc -l -5`: Show last 5 commands.

## Python

*   **`secrets` module:** A replacement for `random` when non-deterministic results are needed.
*   **`re` (regex) module:**
    *   Can be used with or without `re.compile()`.
    *   `re.findall()` finds all instances of a pattern.
    *   Use r-strings (e.g., `r'pattern'`) for literal characters.
    *   `[^a-zA-Z0-9]` is the same as `\W`.
*   **`all()` with list comprehensions:** `if all([constraint <= len(re.findall(pattern, password)) for constraint, pattern in constraints]):`
*   **`input()` vs. `read` (bash):** `input()` is the Python equivalent for getting user input.
*   **Unpacking variables:** Use `*` to unpack a set of variables (e.g., `*variable_name`).
*   **`yield from`:** Useful for delegating to sub-generators.
*   **iPython/Jupyter:**
    *   Pass shell command output to Python variables: `directory_contents = !ls`
    *   Pass Python variables to the shell: `!echo {message}`
    *   Use `%cd` to change directories (automagic, so `cd` often works).
    *   Magic functions: `%cat`, `%cp`, `%env`, `%ls`, `%man`, `%mkdir`, `%more`, `%mv`, `%pwd`, `%rm`, `%rmdir`.
    *   `.grep()` method on shell return objects is powerful.
*   **10 Useful Functions:**
    *   `@dataclass`: Automatically adds `__init__`, `__repr__`, etc.
    *   `pprint`: Pretty-prints nested data structures.
    *   `collections.defaultdict`: Provides a default value for missing keys.
    *   `pickle`: Serialize Python objects to files.
    *   `any()` and `all()`: Check for `True` values in iterables.
    *   `enumerate(start=1)`: Start indexing from 1.
    *   `collections.Counter`: Counts items in an iterable.
    *   `timeit`: Measures execution time of code snippets.
    *   `itertools.chain`: Chains multiple iterators together.
*   **`@staticmethod`:** A regular function defined within a class (for organization).
*   **`@classmethod`:** A method bound to the class, not an instance.
*   **Walrus operator `:=`:** Assign a value to a variable as part of an expression.
    *   `while (line := input("...")) != "quit":`
*   **String formatting:**
    *   `"{0} is {1}".format("truth", "beauty")`
    *   `"Hello, %s" % "Aaron"`
    *   `logging.info("User %s spent $%0.2f", username, amount)`
*   **List Comprehensions:**
    *   With if/else: `[i**2 if i%2==0 else i**3 for i in [1,2,3,4,5]]`
    *   Multiple for loops: `[i for row in mat for i in row if i%2==0]`
    *   Nested: `[[word for word in sentence.split(' ')] for sentence in sentences]`
*   **Lambda Functions:**
    *   Immediate execution: `(lambda x, y: x**y)(3, 5)`
    *   With `filter()`: `list(filter(lambda x: (x % 2 == 0), mylist))`
    *   With `map()`: `list(map(lambda x: x**2, mylist))`
    *   With pandas `apply()`: `df.apply(lambda row: row['First'] * row['Second'], axis=1)`
    *   With pandas `applymap()`: `df.applymap(lambda x: x**2)`
*   **Iterators:** An iterable that remembers its state (has a `__next__()` method).
*   **Generators:** Created with `yield`, which makes a function remember its state.

## Git

*   **Core workflow:** Create a branch, make changes, commit, and merge.
*   **`git init`**: Initialize a new repository.
*   **`git checkout -b <name>`**: Create and switch to a new branch.
*   **`git add <filename>`**: Stage changes for a commit.
*   **`git commit -m 'message'`**: Commit staged changes.
*   **`git log`**: View commit history (`--oneline` for condensed view).
*   **`git diff`**: See changes in uncommitted files.
*   **`git branch`**: List branches (`-d` to delete).
*   **`git merge <branch_name>`**: Merge a branch into the current branch.
*   **`git rebase <branch_name>`**: Update the current branch with commits from another. Use often to maintain a clean commit history.
*   **`git stash`**: Temporarily store changes to work on something else.
    *   `git stash list`, `git stash pop`, `git stash apply`, `git stash drop`
*   **`git reset HEAD~<number>`**: Go back in time, discarding commits (`--soft` vs. `--hard`).
*   **`git revert HEAD`**: Create a new commit that undoes a previous commit.
*   **`.gitignore`**: Add file names to this file to have Git ignore them.
*   **Interactive Rebasing:** `git rebase --interactive HEAD~<number>` to squash, reorder, or edit commits.

## SQL

*   **Regex:** Use `~` for case-sensitive and `~*` for case-insensitive matching. `!~` and `!~*` for negation.
*   **Window Functions:** Perform operations across rows (e.g., `rank()`, `dense_rank()`). Use `OVER (PARTITION BY ...)` to define the window.
*   **`CREATE TABLE`:** Requires `()` after the table name: `CREATE TABLE table_name();`
*   **`JOIN...ON` vs. `USING`:** Use `USING(shared_column_name)` as a shorthand when column names are the same in both tables.

## Data Science & Machine Learning

*   **HDF5 files:** A file format for storing large amounts of numerical data.
*   **Surprise, Entropy, Cross-Entropy, KL Divergence:**
    *   **Entropy:** Average surprise of a probability distribution.
    *   **Cross-Entropy:** Measures the difference between a true distribution (P) and a model's believed distribution (Q).
    *   **KL Divergence:** `H(P,Q) - H(P)`. The extra "surprise" from using Q instead of P. Minimizing KL divergence is equivalent to minimizing cross-entropy in ML.
*   **Kernel Trick (SVMs):** Implicitly maps data to a higher dimension to find a separating hyperplane, without explicitly computing the new coordinates.
*   **Precision vs. Recall:**
    *   **Precision:** How many retrieved items are relevant? (TP / (TP + FP))
    *   **Recall:** How many relevant items are retrieved? (TP / (TP + FN))
*   **Bias-Variance Tradeoff:**
    *   **High Bias:** Underfitting.
    *   **High Variance:** Overfitting.
*   **Box-Cox Transformation:** Transforms data to more closely resemble a normal distribution.
*   **Jaccard Index:** Ratio of the intersection to the union of two sets.
*   **scikit-learn `Pipeline`:** A tool to chain multiple data processing steps together.

## Pandas & Numpy

*   **Pandas:**
    *   Conditional selection: `df[df['col'] > 0]` and `df.loc[df['col'] > 0]`
    *   Combine conditions with `&` (and) and `|` (or).
    *   `set_index()` and `reset_index()`.
    *   Use `inplace=True` to modify a DataFrame directly.
*   **Numpy:**
    *   `.shape` and `.ndim` are attributes, not methods (no `()`).
    *   `np.r_[]`: A quick way to build complex arrays.
    *   `np.lexsort()`: Sort by multiple columns.
    *   `np.vectorize()`: Allows a scalar function to operate on arrays.
    *   `np.digitize()`: Assigns array elements to bins.
    *   `np.argwhere()` or `np.nonzero()`: Find indices of non-zero elements.
    *   **Masks:** Boolean arrays used for indexing. `mask.sum()` counts `True` values.
    *   `np.where(condition, value_if_true, value_if_false)`
    *   `reshape(-1)`: A placeholder for Numpy to automatically calculate a dimension.
    *   `flatten()` creates a copy; `ravel()` creates a view.

## Tools & Editors (VS Code, Chrome, etc.)

*   **VS Code:**
    *   **Interactive Window:** Approximates the Matlab experience.
    *   **Toggle Terminal:** `control` + `~`
    *   **Switch focus to Terminal:** `control` + `command` + `t`
    *   **Switch focus to Editor:** `command` + `1`
    *   **Copy file path:** `option` + `command` + `c`
    *   **Multi-cursor edit:** Select a word, then `command` + `d` to select next occurrences.
    *   **Show hidden files in dialog:** `shift` + `command` + `.`
    *   **File Explorer:** `command` + `p`
    *   **Go to line:** `command` + `p`, then `:` + `<line_number>`
    *   **List symbols in file:** `command` + `shift` + `o`
*   **Chrome:**
    *   **Save all open tabs:** `command` + `shift` + `d`
    *   **Close window (preserve tab groups):** `command` + `shift` + `w`
    *   **Go to address bar:** `command` + `l`
    *   **Search open tabs:** `command` + `shift` + `a`
*   **TextEdit:**
    *   **Go to end of document:** `command` + `down arrow`
    *   **Create URL link:** `command` + `k`
*   **Nano:**
    *   **Toggle line numbers:** `escape` + `n` (or `control` + `c` within file)
    *   **Delete all lines:** `control` + `shift` + `6`, then `control` + `k`
    *   **Undo:** `escape` + `u`
*   **Docker:**
    *   `docker ps`: List running containers.
    *   `docker start -a <id>`: Start and attach to a container.
    *   `docker logs <id>`: View container logs.
    *   `docker exec -it <id> sh`: Get a shell inside a running container.
*   **Conda:**
    *   `conda env export > environment.yml`: Export environment.
    *   `conda env create -f environment.yml`: Create environment from file.
    *   `conda create -n myenv python=3.11 ...`: Create a new environment with specific packages.
    *   `conda env remove -n myenv`: Remove an environment.
    *   `conda create -n new_env --clone old_env`: Clone an environment.
*   **Jupyter Notebooks:**
    *   Clear cell outputs from command line: `jupyter nbconvert --clear-output --inplace your_notebook.ipynb`
    *   `nbstripout`: A tool to strip output from notebooks before committing to Git.
