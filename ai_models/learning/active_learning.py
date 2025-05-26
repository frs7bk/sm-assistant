
class ActiveLearning:
    def __init__(self):
        self.pending_queries = []

    def suggest_clarification(self, input_text, responses):
        self.pending_queries.append((input_text, responses))
        return f"لم أكن متأكدًا، هل تقصد: {', أو '.join(responses)}؟"

    def receive_feedback(self, input_text, selected_response):
        return {"confirmed": True, "best_response": selected_response}
