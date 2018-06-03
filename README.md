Multi Armed Bandit Problem
==========================

This is a basic implementation of the Multi Armed Bandit problem using NumPy explained [here](https://en.wikipedia.org/wiki/Multi-armed_bandit).  

Algorithm used
--------------

- The `test_greedy()` function uses the epsilon-greedy algorithm, running `num_iterations` times per epsilon value

How to run
----------
* Make sure you have Python 3.5 installed
* Clone the repository
* Open directory location in terminal/command prompt
* Run the following commands:
- `pip install virtualenv`
- `virtualenv venv`
- `venv\Scripts\activate`
- `pip install -r requirements.txt`
- `python epsilon_comparisons.py`

Results
-------
![](Epsilon-Greedy%20Algorithm.png?raw=true)