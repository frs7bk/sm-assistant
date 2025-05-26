
class FewShotLearner:
    def __init__(self):
        self.examples = []

    def add_example(self, input_text, response_text):
        self.examples.append((input_text, response_text))

    def generate_with_few_shot(self, input_text):
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.examples[-3:]])
        return f"{context}\nQ: {input_text}\nA:"
