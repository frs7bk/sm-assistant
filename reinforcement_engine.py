
import random
import json
import os

class ReinforcementLearner:
    def __init__(self, memory_path="learning_memory.json"):
        self.memory_path = memory_path
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, "w") as f:
                json.dump([], f)
        with open(self.memory_path, "r") as f:
            self.memory = json.load(f)

    def log_interaction(self, input_text, response_text, reward):
        interaction = {
            "input": input_text,
            "response": response_text,
            "reward": reward
        }
        self.memory.append(interaction)
        with open(self.memory_path, "w") as f:
            json.dump(self.memory[-100:], f)  # Keep last 100 for training

    def improve_response(self, input_text, options):
        best_score = float("-inf")
        best_response = None
        for option in options:
            score = self.evaluate_response(input_text, option)
            if score > best_score:
                best_score = score
                best_response = option
        return best_response

    def evaluate_response(self, input_text, response):
        for entry in reversed(self.memory):
            if entry["input"] == input_text and entry["response"] == response:
                return entry["reward"]
        return random.uniform(0, 0.3)  # Initial randomness if unseen
