# prsample

Inplementation of pseudo-random sampling from a 

# Installation

```sh
pip3 install prsample
```
or
```sh
pip3 install --user prsample
```

# Example usage

The minimum required to 

```
class Example():
    def __init__(self, ... ):
    	# State to describe an example
        return

    def __hash__(self): 
        return super.__hash__((self.class_a, self.a, self.class_b, self.b))

    def __str__(self): 
        return str(self.class_a)  + '(' + str(self.a) + ') ' + str(self.class_b) + '(' +str(self.b) + ')'

    def is_valid(self, class_list):
    	# Optional checks go here
        return True

    def get(self):
        return (self.class_a, self.a, self.class_b, self.b)
```
