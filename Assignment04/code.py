
import random
import numpy as np

# -------------------------------
# 1. Game and learning parameters
# -------------------------------
MAX_ROWS = 6
MAX_COLS = 6
MAX_STEPS = 100
INITIAL_GRID = np.zeros((6,6))
start = (0,0)
goal = (5,5)
obstacles = [(1,1),(1,2),(2,4),(3,1),(4,3)]

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

EPISODES = 100

# # Q[row,column, action] stores the value of an action.
# # Actions are up,down,right,left.
Q = np.zeros((6,6,4))

# # -------------------------------
# # 2. Get legal actions
# # -------------------------------
def take_step(current_state, action):
    row, col = current_state
    
    if(action == 0): # up
        next_row, next_col = row - 1, col
    elif(action == 1): # down
        next_row, next_col = row + 1, col
    elif(action == 2): # right
        next_row, next_col = row, col + 1
    elif(action == 3): # left
        next_row, next_col = row, col - 1

    new_state = (next_row, next_col)

    if next_row < 0 or next_row > MAX_ROWS-1 or next_col < 0 or next_col > MAX_COLS-1:
        return current_state, -10
    elif new_state in obstacles:
        return current_state, -20
    elif new_state == goal:
        return new_state, 100
    else:
        return new_state, -1



# # -------------------------------
# # 4. Train the Q-learning agent
# # -------------------------------
def train_agent():
    epsilon = EPSILON
    rewards_per_episode = []

    for episode in range(EPISODES):
        current_state = start
        episode_reward = 0

        for step in range(MAX_STEPS):
            row, col = current_state

            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q[row, col])

               # Agent selects an action.
            next_state, reward = take_step(current_state, action)
            next_row, next_col = next_state
            episode_reward += reward

            old_value = Q[row, col, action]
            future_max = np.max(Q[next_row, next_col])

            Q[row, col, action] = old_value + ALPHA * (reward + GAMMA * future_max - old_value)
            current_state = next_state

            if current_state == goal:
                break

        rewards_per_episode.append(episode_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

def display_learned_policy():
    policy_grid = np.full((MAX_ROWS, MAX_COLS), ' ')
    for row in range(MAX_ROWS):
        for col in range(MAX_COLS):
            if (row, col) == goal:
                policy_grid[row, col] = 'G'
            elif (row, col) in obstacles:
                policy_grid[row, col] = 'X'
            else:
                best_action = np.argmax(Q[row, col])
                if best_action == 0:
                    policy_grid[row, col] = '↑'
                elif best_action == 1:
                    policy_grid[row, col] = '↓'
                elif best_action == 2:
                    policy_grid[row, col] = '→'
                elif best_action == 3:
                    policy_grid[row, col] = '←'

    print("Learned Policy:")
    for row in policy_grid:
        print(' '.join(row))
    
def final_navigation():
    epsilon = 0
    current_state = start
    reward_total = 0
    final_route = [current_state]
    
    for step in range(MAX_STEPS):
        row, col = current_state
        action = np.argmax(Q[row, col])

        # Agent selects an action.
        next_state, reward = take_step(current_state, action)
        next_row, next_col = next_state
        reward_total += reward
        current_state = next_state
        final_route.append(current_state)

        if current_state == goal:
            break
    
    final_navigation_grid = np.full((MAX_ROWS, MAX_COLS), ' ')
    for row in range(MAX_ROWS):
        for col in range(MAX_COLS):
            if (row, col) == goal:
                final_navigation_grid[row, col] = 'G'
            elif (row, col) in obstacles:
                final_navigation_grid[row, col] = 'X'
            elif (row, col) in final_route:
                final_navigation_grid[row, col] = '*'

    print("\nFinal Navigation Path:")
    for row in final_navigation_grid:
        for cell in row:
            print(cell, end=' ')
        print()

if __name__ == "__main__":
    train_agent()
    display_learned_policy()
    final_navigation()