from agent import Agent
from tictactoe import TicTacToe

from time import sleep
import json


player1 = Agent(prompt='player1.prompt') # has access to memory and is storing
player2 = Agent(prompt='player2.prompt') # has NO access to memory and isn't storing

game = TicTacToe(3)

model_turn = True

while not game.is_game_over():
    if model_turn:
        print(f'--------------------\n{game.get_state()}\n--------------------')

        row, col = player1.submit_board(game.get_state(), game.is_valid_move)["move"] # type: ignore

        prev_state = game.get_state_flat()

        print(row, col)

        success = game.make_move(int(row), int(col))

        if success:
            player1.store_connection(prev_state)
            model_turn = not model_turn
    
    else:
 
        print(f'--------------------\n{game.get_state()}\n--------------------')

        # row, col = player2.submit_board(game.get_state(), game.is_valid_move)["move"]
        row, col = input("Human (R C): ").split(" ")
        print(row, col)

        success = game.make_move(int(row), int(col))

        if success:
            model_turn = not model_turn
    
    sleep(1)

print(f'----------------------------------------')
print(f'--------------------\n{game.get_state()}\n--------------------')
print(f"The winner is: {game.get_winner()}")

player1.game_results.append({
    "winner": game.get_winner(),
    "final_board": game.get_state()
})

# Save player1 stats
with open("player1_stats.json", "w") as f:
    json.dump({
        "memory_accesses": player1.memory_accesses,
        "memory_node_counts": player1.memory_node_counts,
        "game_results": player1.game_results
    }, f, indent=2)



# player1.memory.export_graph()
# player2.memory.export_graph()
