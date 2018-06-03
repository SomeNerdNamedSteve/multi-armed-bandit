from bandit import Bandit, seed_val

import numpy as np
import matplotlib.pyplot as plt

# The following function uses
def test_greedy(epsilon):
    # Problem setup
    max_num_bandits = 101
    num_bandits = np.random.randint(1,max_num_bandits) # set number of bandits
    num_iterations = np.random.randint(10000, 1000000) # set number of iterations
    m_vals = [np.random.randint(1, max_num_bandits) for _ in range(num_bandits)] # create True Mean values
    bandits = [Bandit(m=i) for i in m_vals] # create bandits
    data = np.empty(num_iterations) # create an empty array for output data

    # Epsilon-Greedy algorithm
    for i in range(num_iterations):
        probability = np.random.rand()
        if probability < epsilon:
            bandit = bandits[np.random.choice(num_bandits)]
        else:
            bandit = bandits[np.argmax([bandit.mean for bandit in bandits])]
        
        output = bandit.pull()
        bandit.update(output)
        data[i] = output
    
    # Get cumulative average of all spins
    cumulative_average = np.cumsum(data) / (np.arange(num_iterations) + 1)

    return cumulative_average

if __name__ == '__main__':
    np.random.seed(seed_val) # seed value set

    epsilons = [0.01, 0.05, 0.1]

    for epsilon in epsilons:
        curr_results = test_greedy(epsilon=epsilon)
        plt.plot(curr_results, label='epsilon = {0}'.format(epsilon))
    
    # greedy_epsilon_10_percent = test_greedy(epsilon=0.1)
    # plt.plot(greedy_epsilon_10_percent, label='epsilon = %s' % (epsilon))    
    
    plt.legend()
    plt.xscale('log')
    plt.savefig('Epsilon-Greedy Algorithm.png')