## Connect 4: Background

Connect Four is a two-player connection board game, in which the players choose a color and then take turns dropping colored tokens into a seven-column, six-row vertically suspended grid. The pieces fall straight down, occupying the lowest available space within the column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own tokens.

![Connect4](https://upload.wikimedia.org/wikipedia/commons/a/ad/Connect_Four.gif)

## Reinforcement learning approaches

In this article, we discuss two approaches to create a reinforcement learning agent to play and win the game.

### Deep Q Learning

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Monte Carlo Tree Search

This approach speeds up the learning process significantly compared to the Deep Q Learning approach. Monte Carlo Tree Search (MTCS) excels in situations where the action space is vast. In the case of Connect 4, the action space is 7. As such, to solve Connect 4 with reinforcement learning, a large number of permutations and combinations of the board must be considered. 

There are 4 steps to a MTCS algorithm.

1. **Setup**. Start in the root of the tree (i.e. the state of the board) and pick a child node (i.e. play a move). To pick a child node, use the upper confident bound and use the following expression to identify which nodes have the most expected reward.

    $w/n + c * \sqrt{ln(N)/n}$, where: 
    
    $w$ = number of wins, $n$ = number of times the node is visited, $N$ = number of times the parent node is visited, $c$ = exploration/exploitation factor.

    The above expression is known as the UCB (upper confidence bound) function and is used to select the next move (node) for the game: The node with the highest UCB is selected.
2. 
