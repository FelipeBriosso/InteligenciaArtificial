from agent import Agent
import numpy as np
import copy

class MinimaxTacTixAgent(Agent):
    def __init__(self, env, depth=3):
        self.env = env
        self.depth = depth

    def get_valid_actions(self, board):
        actions = []
        size = board.shape[0]
        for is_row in [0, 1]:
            for idx in range(size):
                line = board[idx, :] if is_row else board[:, idx]
                start = None
                for i in range(size):
                    if line[i] == 1:
                        if start is None:
                            start = i
                    elif start is not None:
                        actions.append([idx, start, i - 1, is_row])
                        start = None
                if start is not None:
                    actions.append([idx, start, size - 1, is_row])
        return actions

    def act(self, obs):
        _, action = self.minimax(obs["board"], self.depth, True, -np.inf, np.inf)
        return action

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        valid_actions = self.get_valid_actions(board)
        if depth == 0 or not valid_actions:
            return self.heuristic_utility(board), None

        best_action = None

        if maximizing_player: # si toca maximizar
            max_eval = -np.inf
            for action in valid_actions:
                new_board = self.simulate_move(board, action)
                eval, _ = self.minimax(new_board, depth - 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else: # si toca minimizar
            min_eval = np.inf
            for action in valid_actions:
                new_board = self.simulate_move(board, action)
                eval, _ = self.minimax(new_board, depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def simulate_move(self, board, action):
        # Realiza una copia del tablero y aplica la acción
        new_board = copy.deepcopy(board)
        row_or_col, start, end, is_row = action
        if is_row:
            new_board[row_or_col, start:end + 1] = 0
        else:
            new_board[start:end + 1, row_or_col] = 0
        return new_board

    def heuristic_utility(self, board):
        # Heurística compuesta: combina varias heurísticas
        return (0 * heuristic_casillas_encendidas(board) +
                0 * heuristic_control(board) +
                0.5 * heuristic_num_jugadas_posibles(board, self) +
                0 * heuristic_control_centro(board) +
                0.5 * heuristic_cantidad_secuencias(board))

def heuristic_casillas_encendidas(board):
    # Heurística simple: cantidad de casillas encendidas
    return np.sum(board)

def heuristic_control(board):
    # Heurística de control: cuenta las secuencias de fichas encendidas
    remaining = board.sum()
    sequences = 0
    size = board.shape[0]

    # Contar secuencias en filas y columnas
    for is_row in [0, 1]:
        for idx in range(size):
            line = board[idx, :] if is_row else board[:, idx]
            in_sequence = False
            for val in line:
                if val == 1:
                    if not in_sequence:
                        sequences += 1
                        in_sequence = True
                else:
                    in_sequence = False

    # Más fichas y menos secuencias es mejor (más control)
    return remaining - sequences * 1.5

def heuristic_num_jugadas_posibles(board, agent):
    return len(agent.get_valid_actions(board))

def heuristic_control_centro(board):
    size = board.shape[0]
    center = size // 2
    value = 0
    for i in range(size):
        for j in range(size):
            if board[i, j] == 1:
                # Penaliza cuanto más lejos del centro
                dist = abs(i - center) + abs(j - center)
                value -= dist
    return value

def heuristic_cantidad_secuencias(board):
    size = board.shape[0]
    sequences = 0
    for is_row in [0, 1]:
        for idx in range(size):
            line = board[idx, :] if is_row else board[:, idx]
            in_seq = False
            for val in line:
                if val == 1:
                    if not in_seq:
                        sequences += 1
                        in_seq = True
                else:
                    in_seq = False
    return -sequences  # penaliza muchas secuencias