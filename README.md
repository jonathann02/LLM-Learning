The model is learning patterns in character transitions (First run)

The first run showed a negative slop (improving) which means it got better at predicting the next characters based on the given characters.

if the model has seen the prefix j o n, it tries to predict what the next character should be (maybe a, maybe end-of-name, etc.). Training pushes it to assign higher probability to the actual next character in the dataset.

When training loss slopes down, it means:

on average, the model is becoming less surprised by the true next character

i.e. it’s getting better at next-character prediction on the training data

