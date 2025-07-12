from minimax_agent import (
    heuristic_casillas_encendidas,
    heuristic_control,
    heuristic_num_jugadas_posibles,
    heuristic_control_centro,
    heuristic_cantidad_secuencias
)

from agent import Agent
import numpy as np
import copy

class ExpectimaxTacTixAgent(Agent):
    def __init__(self, env, depth=3):
        super().__init__(env)  
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
        _, action = self.expectimax(obs["board"], self.depth, True)
        return action

    def expectimax(self, board, depth, maximizing_player):
        valid_actions = self.get_valid_actions(board)
        if depth == 0 or not valid_actions:
            return self.heuristic_utility(board), None

        if maximizing_player:
            max_eval = -np.inf
            best_action = None
            for action in valid_actions:
                new_board = self.simulate_move(board, action)
                eval, _ = self.expectimax(new_board, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
            return max_eval, best_action
        else:
            total = 0
            for action in valid_actions:
                new_board = self.simulate_move(board, action)
                eval, _ = self.expectimax(new_board, depth - 1, True)
                total += eval
            expected_value = total / len(valid_actions)
            return expected_value, None

    def simulate_move(self, board, action):
        new_board = copy.deepcopy(board)
        row_or_col, start, end, is_row = action
        if is_row:
            new_board[row_or_col, start:end + 1] = 0
        else:
            new_board[start:end + 1, row_or_col] = 0
        return new_board

    def heuristic_utility(self, board):
        return (
            0 * heuristic_casillas_encendidas(board) +
            1 * heuristic_control(board) +
            0.5 * heuristic_num_jugadas_posibles(board, self) +
            0 * heuristic_control_centro(board) +
            0.5 * heuristic_cantidad_secuencias(board)
        )
