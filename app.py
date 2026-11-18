import os

import gradio as gr
from transformers import pipeline

# Local folder produced by the notebook, or a Hugging Face Hub id such as "username/bert-intent-ecommerce"
MODEL_ID = os.getenv("MODEL_ID", "bert_intent_model")
# Below this confidence the bot asks the user to rephrase instead of guessing.
# Choose it from the threshold table in the notebook.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# Templated replies. This demo has no backend, so replies offer help and never claim an action was done.
RESPONSES = {
    "contact_customer_service": "I can connect you with customer support. What do you need help with?",
    "review": "Thanks for wanting to share feedback. Please tell me what you'd like to say.",
    "get_refund": "I can help with a refund. Please share your order number.",
    "switch_account": "I can help you switch accounts. Which account would you like to use?",
    "complaint": "I'm sorry to hear that. Please tell me more about the issue so I can help.",
    "check_payment_methods": "You can pay by credit card, UPI or net banking.",
    "contact_human_agent": "I can hand you over to a human agent.",
    "delivery_period": "Deliveries usually take 3 to 5 business days.",
    "check_invoice": "I can help you check your invoice. Which order is it for?",
    "get_invoice": "I can help you get your invoice. Which order is it for?",
    "newsletter_subscription": "I can help you subscribe to our newsletter. Which email address should we use?",
    "place_order": "I can help you place an order. What would you like to order?",
    "check_cancellation_fee": "I can help you check the cancellation fee. Which order is it for?",
    "delete_account": "I can help with deleting your account. This is permanent, so I'll ask you to confirm first.",
    "cancel_order": "I can help you cancel your order. Please share your order number.",
    "payment_issue": "I'm sorry about the payment trouble. Can you tell me what happened?",
    "create_account": "I can help you create an account.",
    "track_refund": "I can help you check your refund status. Please share your order number.",
    "set_up_shipping_address": "I can help you set up a shipping address.",
    "track_order": "I can help you track your order. Please share your order number.",
    "change_order": "I can help you change your order. Please share your order number and what to change.",
    "edit_account": "I can help you update your account information.",
    "check_refund_policy": "I can explain our refund policy. What would you like to know?",
    "delivery_options": "We offer standard and express delivery.",
    "recover_password": "I can help you recover your password.",
    "registration_problems": "I'm sorry you're having trouble registering. What error do you see?",
    "change_shipping_address": "I can help you change your shipping address.",
}
FALLBACK = "I'm not sure how to help with that yet. Could you rephrase?"
LOW_CONFIDENCE = (
    "I'm not sure I understood that. I can help with orders, refunds, payments, "
    "shipping, invoices and accounts. Could you rephrase?"
)

classifier = pipeline("text-classification", model=MODEL_ID, top_k=3)


def respond(message: str):
    message = (message or "").strip()
    if not message:
        return "Please type a question.", {}

    preds = classifier(message, truncation=True, max_length=128)
    if preds and isinstance(preds[0], list):  # some transformers versions nest the result
        preds = preds[0]

    scores = {p["label"]: float(p["score"]) for p in preds}
    top = preds[0]
    if top["score"] < CONFIDENCE_THRESHOLD:
        return LOW_CONFIDENCE, scores
    return RESPONSES.get(top["label"], FALLBACK), scores


demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(lines=2, label="Your message", placeholder="Ask about orders, refunds, accounts..."),
    outputs=[
        gr.Textbox(label="Bot reply"),
        gr.Label(num_top_classes=3, label="Top predicted intents"),
    ],
    title="🛍️ E-commerce Support Chatbot",
    description=(
        "BERT-based intent recognition over 27 customer-support intents. "
        "Demo only: replies are templated and no real actions are performed."
    ),
    examples=[
        ["Where is my order?"],
        ["I want my money back"],
        ["I forgot my password"],
        ["Can I talk to a real person?"],
        ["What's the weather like today?"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
