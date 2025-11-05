# Python Classes: `__init__` vs. `()`

Let's cut to the chase. The two most confusing parts of Python's OOP are:
1.  Where do parameters go?
2.  What are the parentheses in `class Task():` for?

Here's the simple answer.

## `__init__`: For an Object's Data

A class is a blueprint. An object is the thing you build *from* that blueprint.

Parameters that define a *specific object's* data belong in the `__init__` constructor, not the class definition. `__init__` runs every time you create a new object to give it a unique state.

The first argument is always `self` (the instance being created). Everything after that is the data needed to build that one object.

### Example: A `Task`

Every task needs a title and due date. That's the data for the instance.

```python
class Task:
  # __init__ takes the data for a specific task.
  def __init__(self, title, due_date):
    self.title = title
    self.due_date = due_date

# Usage: Pass data to the class to construct an object.
my_task = Task("Finish Report", "2025-11-05")
print(f"Task: {my_task.title}") # -> Task: Finish Report
```

## Class Parentheses: For Inheritance Only

The parentheses in a class definition are **exclusively for inheritance**.

Inheritance creates a new class that is a specialized version of a parent class. The "child" class gets all the features of the "parent."

### Example: An `UrgentTask`

An `UrgentTask` is just a `Task` with a priority. It *inherits* from `Task`.

```python
# Parent Class
class Task:
  def __init__(self, title, due_date):
    self.title = title
    self.due_date = due_date
    self.is_complete = False

  def mark_complete(self):
    self.is_complete = True

# Child Class: UrgentTask IS A Task
class UrgentTask(Task):
  def __init__(self, title, due_date, priority):
    # Call the parent's constructor to handle the shared data.
    super().__init__(title, due_date)
    # Add the new, specialized data.
    self.priority = priority

# Usage
urgent_task = UrgentTask("Deploy security patch", "2025-11-01", "High")
urgent_task.mark_complete() # This method is inherited from Task!
print(f"'{urgent_task.title}' is complete.") # -> 'Deploy security patch' is complete.
```

## Summary: Where to Put Things

| Location | What Goes There? | Example | Purpose |
| :--- | :--- | :--- | :--- |
| **`__init__`** | Instance Data | `def __init__(self, title):` | Build a specific object with its unique state. |
| **Class `()`** | Parent Class | `class Child(Parent):` | Inherit features from another class. |

## Practical Patterns

### 1. Defaults with Class Variables

Use class variables for data shared by *all* instances. Use instance variables (`self.x`) for data unique to *one* instance.

```python
class DatabaseConnection:
  # CLASS VARIABLE: A default shared by all instances.
  DEFAULT_PORT = 5432

  def __init__(self, host, db_name, user, port=None):
    # INSTANCE VARIABLES: Unique to this connection.
    self.host = host
    self.db_name = db_name
    self.user = user
    self.port = port or self.DEFAULT_PORT # Fallback to default

  def connect(self):
    print(f"Connecting to {self.db_name} on {self.host}:{self.port}...")

# Usage
prod_db = DatabaseConnection("prod.server.com", "MainDB", "admin")
dev_db = DatabaseConnection("localhost", "TestDB", "dev", port=1433)

prod_db.connect() # -> ... on prod.server.com:5432
dev_db.connect()  # -> ... on localhost:1433
```

### 2. Composition: Building Objects from Other Objects

An object's data can be other objects. Just pass them into the constructor like any other parameter. This is called composition.

```python
class Item:
  def __init__(self, name, price):
    self.name = name
    self.price = price

class ShoppingCart:
  def __init__(self, owner_name):
    self.owner_name = owner_name
    self.items = [] # This will hold Item objects

  def add_item(self, item: Item):
    self.items.append(item)

  def get_total(self):
    return sum(item.price for item in self.items)

# Usage
laptop = Item("Laptop Pro", 1200.00)
mouse = Item("Wireless Mouse", 25.00)

# Create the container and pass in the component objects.
my_cart = ShoppingCart("Alice")
my_cart.add_item(laptop)
my_cart.add_item(mouse)

print(f"Total for {my_cart.owner_name}: ${my_cart.get_total():.2f}") # -> Total for Alice: $1225.00
```

Master this distinction—`__init__` for data, `()` for inheritance—and you've mastered the core of Python OOP.
