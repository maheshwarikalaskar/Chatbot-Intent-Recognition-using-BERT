# E-commerce Intent Recognition with BERT

Fine-tuned `bert-base-uncased` that classifies customer-support queries into 27 intents, wrapped in a Gradio chatbot with a confidence threshold and a fallback reply.

## What it does

A user types a message such as "Where is my order?" and the model predicts the intent (`track_order`). The app replies with a templated answer, shows the top 3 predicted intents with scores, and asks the user to rephrase when the model is not confident enough.

## Evaluation

`Model Training.ipynb` trains and evaluates the model end to end. When you run it, it:

- removes duplicate queries and checks for conflicting labels
- makes a stratified 80/10/10 train/validation/test split
- selects the best epoch on validation macro-F1, never on the test set
- reports test accuracy, macro-F1, a per-intent classification report and a normalised confusion matrix
- lists the most confused intent pairs and example misclassifications
- measures how many test queries have a near-duplicate in the training set
- tabulates accuracy against confidence threshold and probes the model with off-topic queries
- writes the headline numbers to `metrics.json`

## Intents

The model predicts one of 27 intents:

`cancel_order`, `change_order`, `change_shipping_address`, `check_cancellation_fee`, `check_invoice`, `check_payment_methods`, `check_refund_policy`, `complaint`, `contact_customer_service`, `contact_human_agent`, `create_account`, `delete_account`, `delivery_options`, `delivery_period`, `edit_account`, `get_invoice`, `get_refund`, `newsletter_subscription`, `payment_issue`, `place_order`, `recover_password`, `registration_problems`, `review`, `set_up_shipping_address`, `switch_account`, `track_order`, `track_refund`

## Dataset

`Customer_Service_Training_Dataset.csv` contains 8,175 customer queries labelled with 27 intents (before de-duplication). Datasets of this kind are often built from templates with paraphrase variation, so scores on them can be higher than on messages typed by real customers. The notebook's near-duplicate check shows how much this matters here.

## Run it

**1. Train** (Colab with a GPU runtime is the easiest option): open `Model Training.ipynb`, upload `Customer_Service_Training_Dataset.csv` next to it, and run all cells. This creates a `bert_intent_model/` folder with the model and tokenizer.

**2. Chat:**

```bash
pip install -r requirements.txt
python app.py
```

The app loads `bert_intent_model/` by default. To load a model from the Hugging Face Hub instead, set `MODEL_ID`:

```bash
MODEL_ID=your-username/your-model python app.py
```

Set `CONFIDENCE_THRESHOLD` (default 0.6) to make the bot more or less willing to answer. The notebook's threshold table shows the trade-off.

Model weights are about 420 MB, so they are not stored in this repository.

## Files

```
.
├── Customer_Service_Training_Dataset.csv
├── Model Training.ipynb    # clean, evaluate, train, save
├── app.py                  # Gradio chatbot
├── requirements.txt
└── README.md
```

## Limitations

- The bot classifies intent and returns templated replies. It is not connected to orders, accounts or payments, so it never performs an action.
- One intent per message. No dialogue state, multi-intent handling or entity extraction (order numbers, dates).
- English only, and a closed set of 27 intents, so off-topic messages can only be caught by the confidence threshold.
- Accuracy on real customer messages is unknown until the model is tested on real data.

## Future work

- Evaluate on real, human-written support messages
- Add an explicit "out of scope" class
- Entity extraction and multi-turn dialogue
- Compare against a distilled model for faster CPU inference

## Author

**Maheshwari Kalaskar**: [GitHub](https://github.com/maheshwarikalaskar) · [LinkedIn](https://www.linkedin.com/in/maheshwarikalaskar/)
