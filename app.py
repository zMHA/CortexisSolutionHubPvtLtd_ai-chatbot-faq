import gradio as gr
import json
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAQs
with open("faqs.json", "r") as f:
    faqs = json.load(f)

# Prepare questions and answers
questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Chatbot logic (FIXED)
def chatbot_response(user_input, history):
    if not user_input.strip():
        return "Please enter a question."

    query_emb = model.encode(user_input, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(query_emb, question_embeddings)[0]
    best_match_idx = int(similarity_scores.argmax())
    best_score = float(similarity_scores[best_match_idx])

    if best_score > 0.6:
        return f"**Answer:** {answers[best_match_idx]}\n\n*(Matched FAQ: {questions[best_match_idx]} — Confidence: {best_score:.2f})*"
    else:
        return "I'm not sure about that. Would you like me to connect you to an admin?"

# Gradio Interface
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="🤖 AI Chatbot for Business FAQs",
    description="Ask any question about the company's products, services, or policies.",
    examples=[
        ["What are your working hours?"],
        ["Do you offer international shipping?"],
        ["Where are you located?"],
        ["How can I contact customer support?"]
    ]
)

if __name__ == "__main__":
    demo.launch()
