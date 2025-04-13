class TicTacToe:
    def __init__(self, board_size=3):
        """Initialize the TicTacToe game with variable board size
        
        Args:
            board_size: Size of the board (e.g., 3 for 3x3, 4 for 4x4, etc.)
        """
        # Validate board size
        if board_size not in [3, 4, 5, 6, 9]:
            raise ValueError("Board size must be 3, 4, 5, 6, or 9")
        
        self.board_size = board_size
        # Initialize empty board
        self.board = [['-' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'  # X goes first
        self.moves = 0
        self.game_over = False
        self.winner = None
        
        # For win condition: number of marks in a line needed to win
        # For 3x3, need 3; for 4x4-6x6, need 4; for 9x9, need 5
        if board_size <= 3:
            self.win_length = 3
        elif board_size <= 6:
            self.win_length = 4
        else:  # 9x9
            self.win_length = 5
    
    def reset(self):
        """Reset the game to initial state"""
        # Keep the same board size, but reset everything else
        board_size = self.board_size
        self.__init__(board_size)
        return self.get_state()
    
    def get_state(self):
        """Return current board state as a string"""
        return '\n'.join([''.join(row) for row in self.board])
    
    def get_state_flat(self):
        """Return current board state as a flattened string"""
        return ''.join([''.join(row) for row in self.board])
    
    def get_board_size(self):
        """Return the size of the board (e.g., 3 for 3x3)"""
        return self.board_size
    
    def is_valid_move(self, row, col):
        """Check if a move is valid"""
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        return self.board[row][col] == '-'
    
    def make_move(self, row, col):
        """Make a move and return if successful"""
        if self.game_over or not self.is_valid_move(row, col):
            return False
        
        self.board[row][col] = self.current_player
        self.moves += 1
        
        # Check for win or draw
        if self._check_win():
            self.game_over = True
            self.winner = self.current_player
        elif self.moves == self.board_size * self.board_size:  # Board is full
            self.game_over = True
            self.winner = None  # Draw
        
        # Switch player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return True
    
    def _check_win(self):
        """Check if current player has won"""
        player = self.current_player
        board = self.board
        n = self.board_size
        win_len = self.win_length
        
        # Check rows
        for row in range(n):
            for col in range(n - win_len + 1):
                if all(board[row][col+i] == player for i in range(win_len)):
                    return True
        
        # Check columns
        for col in range(n):
            for row in range(n - win_len + 1):
                if all(board[row+i][col] == player for i in range(win_len)):
                    return True
        
        # Check diagonals (top-left to bottom-right)
        for row in range(n - win_len + 1):
            for col in range(n - win_len + 1):
                if all(board[row+i][col+i] == player for i in range(win_len)):
                    return True
        
        # Check diagonals (top-right to bottom-left)
        for row in range(n - win_len + 1):
            for col in range(win_len - 1, n):
                if all(board[row+i][col-i] == player for i in range(win_len)):
                    return True
        
        return False
    
    def get_legal_moves(self):
        """Return list of legal moves as (row, col) tuples"""
        if self.game_over:
            return []
        
        moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == '-':
                    moves.append((row, col))
        return moves
    
    def get_winner(self):
        """Return the winner ('X', 'O', or None for draw)"""
        return self.winner
    
    def is_game_over(self):
        """Return if the game is over"""
        return self.game_over
    
    def __str__(self):
        """String representation of the board"""
        return '\n'.join(['|'.join(row) for row in self.board]) 