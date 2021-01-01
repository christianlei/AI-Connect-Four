import numpy as np


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.opponent_number = int(2 / player_number)
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.possible_moves = (0, 1, 2, 3, 4, 5, 6)
        self.depth = 4
        self.board = None

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        alpha = float("-inf")
        beta = float("inf")
        v_max = float("-inf")
        self.board = board
        possible_moves = self.get_possible_moves()
        col_to_return = np.random.choice(possible_moves)
        for move in possible_moves:
            self.update_board(move, self.player_number)
            v = self.get_alpha_beta_min_value(self.depth, alpha, beta)
            self.revert_move(move)
            if v > v_max:
                v_max = v
                col_to_return = move
        return col_to_return

    def get_alpha_beta_max_value(self, depth, alpha, beta):
        v_max = float("-inf")
        possible_moves = self.get_possible_moves()

        if self.cutoff_function(depth, possible_moves):
            return self.evaluation_function(self.player_number, self.opponent_number)

        for move in possible_moves:
            if depth > 0:
                self.update_board(move, self.player_number)
                v_max = max(v_max, self.get_alpha_beta_min_value(depth - 1, alpha, beta))
                self.revert_move(move)
                if v_max >= beta:
                    return v_max
                alpha = max(alpha, v_max)
        return v_max

    def get_alpha_beta_min_value(self, depth, alpha, beta):
        v_min = float("inf")
        possible_moves = self.get_possible_moves()

        if self.cutoff_function(depth, possible_moves):
            v = self.evaluation_function(self.player_number, self.opponent_number)
            return v

        for move in possible_moves:
            if depth > 0:
                self.update_board(move, self.opponent_number)
                v_min = min(v_min, self.get_alpha_beta_max_value(depth - 1, alpha, beta))
                self.revert_move(move)
                if v_min <= alpha:
                    return v_min
                beta = min(beta, v_min)
        return v_min

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        self.board = board
        v_max = float("-inf")
        possible_moves = self.get_possible_moves()
        col_to_return = np.random.choice(possible_moves)
        for move in possible_moves:
            self.update_board(move, self.player_number)
            v = self.get_expectimax_chance_value(self.depth - 1)
            if v > v_max:
                v_max = v
                col_to_return = move
        return col_to_return

    def get_expectimax_max_value(self, depth):
        v_max = float("-inf")

        possible_moves = self.get_possible_moves()
        if self.cutoff_function(depth, possible_moves):
            return self.evaluation_function(self.player_number, self.opponent_number)

        for move in possible_moves:
            self.update_board(move, self.player_number)
            v = self.get_expectimax_chance_value(depth - 1)
            self.revert_move(move)
            v_max = max(v_max, v)
        return v_max

    def get_expectimax_chance_value(self, depth):
        v_chance = 0
        possible_moves = self.get_possible_moves()
        probability = 1 / len(possible_moves)

        if self.cutoff_function(depth, possible_moves):
            return self.evaluation_function(self.player_number, self.opponent_number)

        for move in self.possible_moves:
            self.update_board(move, self.opponent_number)
            v = self.get_expectimax_max_value(depth - 1)
            self.revert_move(move)
            v_chance += v * probability
        return v_chance

    def cutoff_function(self, depth, possible_moves):
        if depth == 0:
            return True
        if len(possible_moves) == 0:
            return True
        if self.is_four_in_a_row():
            return True
        return False

    def evaluation_function(self, player, opponent):
        """
        Given the current stat of the board, return the scalar value that
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        two_in_a_row = '{0}{0}00'.format(player)
        two_in_a_row_enemy = '{0}{0}00'.format(opponent)

        three_in_a_row = '{0}{0}{0}0'.format(player)
        three_in_a_row_enemy = '{0}{0}{0}0'.format(opponent)

        four_in_a_row = '{0}{0}{0}{0}'.format(player)
        four_in_a_row_enemy = '{0}{0}{0}{0}'.format(opponent)

        did_opponent_win = self.is_sequence_in_board(four_in_a_row_enemy)
        did_i_win = self.is_sequence_in_board(four_in_a_row)

        player_score = self.is_sequence_in_board(
            three_in_a_row) * 1000 + self.is_sequence_in_board(
            two_in_a_row) * 100
        opponent_score = self.is_sequence_in_board(
            three_in_a_row_enemy) * 1000 + self.is_sequence_in_board(
            two_in_a_row_enemy) * 100

        if did_opponent_win:
            return float("-inf")
        elif did_i_win:
            return float("inf")

        return player_score - opponent_score

    def revert_move(self, move):
        zero_count = 0
        if 0 in self.board[:, move]:
            for i in self.board[:, move]:
                if i == 0:
                    zero_count += 1
            row = zero_count
            self.board[row, move] = 0
        else:
            self.board[0, move] = 0

    def update_board(self, move, player_num):
        if 0 in self.board[:, move]:
            for row in range(1, self.board.shape[0]):
                update_row = -1
                if self.board[row, move] > 0 and self.board[row - 1, move] == 0:
                    update_row = row - 1
                elif row == self.board.shape[0] - 1 and self.board[row, move] == 0:
                    update_row = row
                if update_row >= 0:
                    self.board[update_row, move] = player_num
                    break

    def get_possible_moves(self):
        possible_moves = []
        for move in self.possible_moves:
            if not self.is_col_full(move):
                possible_moves.append(move)
        return possible_moves

    def is_col_full(self, move):
        open_space = '0'
        board_transpose = self.board.T
        to_str = lambda a: ''.join(a.astype(str))

        if open_space in to_str(board_transpose[move]):
            return False
        return True

    def is_four_in_a_row(self):
        player_1_win_str = '1111'
        player_2_win_str = '2222'
        if self.is_sequence_in_board(player_1_win_str) or self.is_sequence_in_board(player_2_win_str):
            return True
        return False

    def is_sequence_in_board(self, sequence_to_find):
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b, player_string):
            for row in b:
                if player_string in to_str(row):
                    return True
            return False

        def check_vertical(player_string):
            return check_horizontal(self.board.T, player_string)

        def check_diagonal(player_string):
            for op in [None, np.fliplr]:
                op_board = op(self.board) if op else self.board

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_string in to_str(root_diag):
                    return True

                for i in range(1, self.board.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_string in diag:
                            return True
            return False

        def is_sequence_in_board(player_string):
            horizontal = check_horizontal(self.board, player_string)
            vertical = check_vertical(player_string)
            diagonal = check_diagonal(player_string)
            if horizontal or vertical or diagonal:
                return True
            return False

        if is_sequence_in_board(sequence_to_find):
            return 1
        return 0


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
