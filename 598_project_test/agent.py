import json
import pickle
from typing import Iterable, List

import torch

from graph_memory import GraphMemory
from vector_memory import VectorizedMemory
from autoencoder import AutoEncoder
import openai

API_KEY = "REPLACE THIS" # TODO REPLACE
MAX_TRIES = 3

class Agent:
    def __init__(self, memory_dir: str = "", prompt: str = "test_prompt.txt"):
        self.ae = AutoEncoder()
        self.ae.load_state_dict(torch.load('autoencoder.model', weights_only=True))

        self.ae.eval()
        self.memory = VectorizedMemory(self.ae)
        
        self.memory_accesses = 0
        self.memory_node_counts = []
        self.game_results = []  # store (winner, final board)

        
        with open('functions.json', 'r') as f:
            self.funcs = json.loads(f.read())
        
        
        openai.api_key = API_KEY
        
        with open(prompt, 'r') as f:
            system_prompt = f.read()

        self.message_hist = [
            {
                "role" : "system",
                "content" : system_prompt
            }
        ]
        
        self.responses = []
    
    def submit_board(self, state: str, validator):
        messages = self.message_hist.copy()
        
        messages.append(
            {
                "role" : "user",
                "content": f"Current board state:\n{state}\n\nRetrieve or make a move."
            }
        )
        
        success, data = self._submit_request(messages)
        num_tries = 0
        while not success and num_tries < MAX_TRIES:
            messages.append(
                {
                    "role" : "user",
                    "content" : f"Here is the retrieved data:\n{data}"
                }
            )
            success, data = self._submit_request(messages)

            num_tries += 1
        
        if not success:
            messages.append(
                {
                    "role" : "user",
                    "content" : "You have retrieved too many times. Please use the make_move."
                }
            )
            success, data = self._submit_request(messages)
            if not success:
                raise Exception(f"Failed to get move from model:\n\t{data}\n\t{messages}")
        
        num_tries = 0
        while not success and not validator(data[0], data[1]) and num_tries < MAX_TRIES:
            messages.append(
                {
                    "role" : "user",
                    "content" : f"Invalid move. Please pick a valid one."
                }
            )
            success, data = self._submit_request(messages)
            num_tries += 1
        
        if not success:
                raise Exception(f"Failed to get move from model:\n\t{data}\n\t{messages}")
        return data

    def store_connection(self, prev_state):
        messages = self.message_hist.copy()
        messages.append(
            {
                "role":"user",
                "content": f"Store the board state ({prev_state}) via store(metadata), where the metadata are any strategies you would reference later."
            }
        )
        
        response = self._get_api_response(messages)
        message = response.choices[0].message.model_dump()
        
        print(f"Model Storage Thought: {message["content"]}")
        
        self._handle_storage_function_call(message, prev_state)

    def _submit_request(self, messages):
        response = self._get_api_response(messages)
        self.responses.append(response)
        response_message = response.choices[0].message.model_dump()
        success, data = self._handle_on_move_function_call(response_message)
        return success, data
    
    def _get_api_response(self, messages):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            functions=self.funcs,
            function_call="auto",
            temperature=0.7,
            messages=messages # type: ignore
        ) # type: ignore

        return response
        
    def _handle_on_move_function_call(self, message):
        if message.get("function_call"):
            print("Model called function:")
            function_name = message["function_call"]["name"]
            arguments = message["function_call"]["arguments"]

            # Parse the arguments
            args = json.loads(arguments)

            print(f"--------------\n{function_name}(\n\t{json.dumps(args,indent=2)}\n)\n--------------")
            # Depending on the function, call your backend
            if function_name == "retrieve":
                try:
                    transitions = self.memory.retrieve(args["state"])
                except:
                    return False, []
                self.memory_accesses += 1
                print("Retrieved transitions:", transitions)
                return False, transitions

            elif function_name == "make_move":
                print("Model made a move:", args)
                return True, args

        return False, {}
    
    def _handle_storage_function_call(self, message, state):
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            arguments = message["function_call"]["arguments"]

            args = json.loads(arguments)
            print(f"--------------\n{function_name}(\n\t{json.dumps(args,indent=2)}\n)\n--------------")
            if function_name == "store":
                print(f"Storing Prev State: {state}")
                self.memory.store(
                    state,
                    metadata=args.get("metadata", "")
                )

                self.memory_node_counts.append(len(self.memory.memory_vectors))

        
        
        