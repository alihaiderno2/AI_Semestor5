Prerequisites:

- Python 3.x

- Pillow library (for generating output images). If you do not have it installed, run: `pip install Pillow`

How to Run: Execute the script from the command line by providing the Python file, the maze text file, and the desired search algorithm (`bfs`, `greedy`, or `astar`) as arguments.

Command Format:

`python maze.py <maze_file.txt> <algorithm>`

Examples:

- Breadth-First Search: `python maze.py maze1.txt bfs`

- Greedy Best-First Search: `python maze.py maze1.txt greedy`

- A\* Search: `python maze.py maze1.txt astar`

<span>**Outputs:** The program will output the text-based maze layout, the solution path, the number of states explored, path cost (moves), maximum frontier size, and runtime directly in the terminal. It will also generate a PNG image file (e.g., </span>`maze1.txtastar.png`<span>) showing the explored states and final path.</span>