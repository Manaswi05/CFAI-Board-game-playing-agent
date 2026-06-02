import math
import copy

# ============================================================
# PEAS MODEL (for Board Game Agent)
# ------------------------------------------------------------
# Performance: Win the game, avoid loss, minimize moves
# Environment: 3x3 Tic-Tac-Toe board
# Actuators: Place X or O on board
# Sensors: Current board state
# ============================================================

PLAYER_X = "X"
PLAYER_O = "O"
EMPTY = " "

# ============================================================
# STATE REPRESENTATION (3x3 grid)
# ============================================================

class TicTacToe:
    def __init__(self):
        self.board = [[EMPTY for _ in range(3)] for _ in range(3)]

    def actions(self):
        return [(r, c)
                for r in range(3)
                for c in range(3)
                if self.board[r][c] == EMPTY]

    def result(self, action, player):
        new_state = copy.deepcopy(self)
        r, c = action
        new_state.board[r][c] = player
        return new_state

    def is_winner(self, player):
        b = self.board

        # rows, cols, diagonals
        for i in range(3):
            if all(b[i][j] == player for j in range(3)):
                return True
            if all(b[j][i] == player for j in range(3)):
                return True

        if all(b[i][i] == player for i in range(3)):
            return True
        if all(b[i][2 - i] == player for i in range(3)):
            return True

        return False

    def terminal(self):
        return self.is_winner(PLAYER_X) or self.is_winner(PLAYER_O) or len(self.actions()) == 0

    def print_board(self):
        for row in self.board:
            print(row)
        print()

# ============================================================
# HEURISTIC EVALUATION FUNCTION
# (used in Minimax depth limiting / greedy strategy)
# ============================================================

def heuristic(state):
    if state.is_winner(PLAYER_X):
        return 10
    if state.is_winner(PLAYER_O):
        return -10

    # small heuristic: center control + corners
    score = 0
    center = state.board[1][1]

    if center == PLAYER_X:
        score += 1
    elif center == PLAYER_O:
        score -= 1

    return score

# ============================================================
# MINIMAX WITH ALPHA-BETA PRUNING
# ============================================================

def minimax(state, depth, alpha, beta, maximizing):
    if state.terminal() or depth == 0:
        return heuristic(state)

    if maximizing:
        max_eval = -math.inf
        for action in state.actions():
            child = state.result(action, PLAYER_X)
            eval = minimax(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # alpha-beta pruning
        return max_eval

    else:
        min_eval = math.inf
        for action in state.actions():
            child = state.result(action, PLAYER_O)
            eval = minimax(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# ============================================================
# BEST MOVE SELECTION (AI AGENT DECISION FUNCTION)
# ============================================================

def best_move(state, player):
    best_score = -math.inf if player == PLAYER_X else math.inf
    best_action = None

    for action in state.actions():
        new_state = state.result(action, player)
        score = minimax(new_state, 3, -math.inf, math.inf, player == PLAYER_O)

        if player == PLAYER_X and score > best_score:
            best_score = score
            best_action = action

        if player == PLAYER_O and score < best_score:
            best_score = score
            best_action = action

    return best_action

# ============================================================
# BFS EXAMPLE (STATE SPACE EXPLORATION IDEA)
# (Not used for playing, but for learning search concepts)
# ============================================================

from collections import deque

def bfs_win_path(start_state):
    queue = deque([(start_state, [])])
    visited = set()

    while queue:
        state, path = queue.popleft()

        if state.is_winner(PLAYER_X):
            return path

        state_key = str(state.board)
        if state_key in visited:
            continue
        visited.add(state_key)

        for action in state.actions():
            next_state = state.result(action, PLAYER_X)
            queue.append((next_state, path + [action]))

    return None

# ============================================================
# SIMPLE GAME LOOP (AI vs AI)
# ============================================================

def play_game():
    game = TicTacToe()
    current_player = PLAYER_X

    while not game.terminal():
        game.print_board()

        if current_player == PLAYER_X:
            move = best_move(game, PLAYER_X)
        else:
            move = best_move(game, PLAYER_O)

        print(f"{current_player} plays: {move}")
        game = game.result(move, current_player)

        current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

    game.print_board()

    if game.is_winner(PLAYER_X):
        print("X wins!")
    elif game.is_winner(PLAYER_O):
        print("O wins!")
    else:
        print("Draw!")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    play_game()