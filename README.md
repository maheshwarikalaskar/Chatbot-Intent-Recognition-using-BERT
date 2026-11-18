# E-commerce Intent Recognition with BERT

Fine-tuned `bert-base-uncased` that classifies customer-support queries into 27 intents, wrapped in a Gradio chatbot with a confidence threshold and fallback.

**Live demo:** _add your Hugging Face Spaces link here_ · **Model:** _add your Hugging Face Hub model link here_

![Demo screenshot](assets/demo.png)

## Results

Evaluated on a held-out, stratified test set that the model never saw during training or model selection (80/10/10 train/validation/test split, `seed=42`). Run `notebooks/Model_Training.ipynb` and paste its output here.

| Metric | Value |
|---|---|
| Test accuracy | _fill in_ |
| Test macro-F1 | _fill in_ |
| Test queries | _fill in_ |
| Test queries with a near-duplicate in train (cosine ≥ 0.9) | _fill in_ |

Confidence threshold trade-off on the test set (the app uses a threshold to avoid answering queries it is unsure about):

| Threshold | Share of queries answered | Accuracy on answered |
|---|---|---|
| 0.5 | _fill in_ | _fill in_ |
| 0.7 | _fill in_ | _fill in_ |
| 0.9 | _fill in_ | _fill in_ |

Most-confused intent pairs and misclassified examples are in the notebook's error-analysis section.

## Intents

The model predicts one of 27 intents:

`cancel_order`, `change_order`, `change_shipping_address`, `check_cancellation_fee`, `check_invoice`, `check_payment_methods`, `check_refund_policy`, `complaint`, `contact_customer_service`, `contact_human_agent`, `create_account`, `delete_account`, `delivery_options`, `delivery_period`, `edit_account`, `get_invoice`, `get_refund`, `newsletter_subscription`, `payment_issue`, `place_order`, `recover_password`, `registration_problems`, `review`, `set_up_shipping_address`, `switch_account`, `track_order`, `track_refund`

## Dataset

_State the source here._ The CSV's column layout matches the public Bitext customer-support dataset. If that is where it comes from, cite it and check its license. _Report the row count after de-duplication._

Queries in datasets like this are often generated from templates with paraphrase variation. Scores on such data show that the pipeline works but are likely higher than on messages typed by real customers. The notebook measures how many test queries have a near-duplicate in the training set so you can see this directly.

## Approach

1. Clean: drop empty and duplicate queries, check for conflicting labels.
2. Split: stratified train/validation/test, so every intent appears in each split.
3. Fine-tune `bert-base-uncased` with the Hugging Face `Trainer`: dynamic padding, best checkpoint selected on validation macro-F1.
4. Evaluate on the untouched test set: accuracy, macro-F1, per-intent report, confusion matrix, error analysis.
5. Calibrate: look at accuracy versus confidence and probe with off-topic queries to choose the app's threshold.
6. Serve: Gradio app that returns a templated reply, shows the top 3 intents with scores, and asks the user to rephrase when confidence is below the threshold.

## Run locally

```bash
git clone https://github.com/maheshwarikalaskar/Chatbot-Intent-Recognition-using-BERT.git
cd Chatbot-Intent-Recognition-using-BERT
pip install -r requirements.txt

# Use the published model...
MODEL_ID=your-username/bert-intent-ecommerce python app.py
# ...or a local folder produced by the notebook
MODEL_ID=bert_intent_model python app.py
```

Model weights are about 420 MB, so they live on the Hugging Face Hub instead of this repository.

To retrain, open `notebooks/Model_Training.ipynb` in Colab (GPU runtime) or locally and run all cells.

## Project structure

```
.
├── app.py                              # Gradio chatbot
├── data/Customer_Service_Training_Dataset.csv
├── notebooks/Model_Training.ipynb      # train + evaluate + save
├── metrics.json                        # test metrics written by the notebook
├── requirements.txt
└── README.md
```

## Limitations

- The bot classifies intent and returns templated replies. It does not connect to orders, accounts or payments, so it never performs any action.
- One intent per message: no dialogue state, no multi-intent handling, no entity extraction (order numbers, dates).
- English only. Trained on a closed set of 27 intents, so out-of-scope messages can only be caught by the confidence threshold.
- Accuracy on real customer messages is unknown until it is tested on real data.

## Future work

- Evaluate on real, human-written support messages
- Add an explicit "out of scope" class
- Entity extraction (order IDs) and multi-turn dialogue
- Compare against a smaller distilled model for faster CPU inference

## Author

**Maheshwari Kalaskar**: [GitHub](https://github.com/maheshwarikalaskar) · [LinkedIn](https://www.linkedin.com/in/maheshwarikalaskar/)
