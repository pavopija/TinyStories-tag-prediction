# TinyStories-Tag-Prediction  
A lightweight LLM-based model using **Hugging Face Transformers** and **PyTorch** to predict narrative attributes such as *“conflict”* and *“good ending”* in the TinyStory dataset.

---

## 📘 Introduction  
We chose Track 1 among the proposed options: our task is to **predict the tags** of a story belonging to the **TinyStories** dataset.  
Each example in the dataset consists of a story and its corresponding features.

We first split the training set into **training** and **validation** subsets. Then we defined a function that creates a **label vector** encoding the tags — a vector of length 6, where each position corresponds to one of the six possible tags. Each entry is set to 1 if the story belongs to that tag, and 0 otherwise.

> Each story may have more than one tag, so this is a **multi-label classification** problem.

We use two different models:
1. **Custom Transformer model** — built from scratch and trained on a subset of the dataset.  
2. **Pretrained model** — fine-tuned from Hugging Face’s `distilbert-base-uncased`.

For each model we use different tokenization strategies and hyperparameters, explained below.

**Team:**  
- *Leonardo Cittadini* – custom model  
- *Laura Cangiotti* and *Pavao Jancijev* – pretrained model  

---

## 🧩 Custom Model  

### Tokenization and Preprocessing  
We built a **vocabulary** using all words and punctuation in the training set (≈ 36 k tokens) plus a `"PAD"` token.  
Key preprocessing decisions:
- All text was **lowercased** (case is not meaningful for this task).  
- **Punctuation** was kept (e.g., quotation marks help detect dialogue).  
- Stories were padded/truncated to a fixed length of **256 tokens**.  
- Unknown tokens in validation/test sets were replaced with `"PAD"`.

### Architecture  
- **Embedding layer** (ignores padding tokens)  
- **3 Transformer layers** with 8 attention heads  
- Hidden dimension slightly smaller than embedding size (faster training, no loss of accuracy)  
- **Max pooling** for representation aggregation (better than mean pooling in this setup)

### Training  
- 250 000 stories  
- 10 epochs, batch size = 64  
- Best validation loss observed at **epoch 4** (after which overfitting begins)

| Epoch | Validation Loss | Training Loss | Accuracy |
|:------|:----------------|:---------------|:----------|
| 1 | 0.1764 | 0.1948 | 0.9328 |
| 2 | 0.1687 | 0.1571 | 0.9349 |
| 3 | 0.1633 | 0.1444 | 0.9370 |
| **4** | **0.1557** | **0.1343** | **0.9402** |
| 5 | 0.1722 | 0.1248 | 0.9317 |
| … | … | … | … |

**Validation Accuracy:** ≈ 94%  
**Test Accuracy (per label):** 0.9418  

**F1 Scores (test set):**

| Label | F1 Score |
|:------|:----------|
| BadEnding | 0.8727 |
| Conflict | 0.4121 |
| Dialogue | 0.9225 |
| Foreshadowing | 0.4195 |
| MoralValue | 0.8460 |
| Twist | 0.8652 |

---

## 🤗 Pretrained Model — DistilBERT  

We fine-tuned **`distilbert-base-uncased`**, a compact version of BERT that does not distinguish uppercase/lowercase letters.  
Its vocabulary (~30 k tokens) is smaller because it tokenizes into **subwords**.

### Training Setup  
- 7 epochs, batch size = 64  
- Best validation loss at **epoch 3**

| Epoch | Validation Loss | Training Loss | Accuracy |
|:------|:----------------|:---------------|:----------|
| 1 | 0.1407 | 0.1688 | 0.9438 |
| 2 | 0.1353 | 0.1245 | 0.9491 |
| **3** | **0.1308** | **0.1056** | **0.9503** |
| 4 | 0.1390 | 0.0876 | 0.9498 |
| 5 | 0.1547 | 0.0709 | 0.9468 |

**Test Accuracy:** 0.9520  

**F1 Scores (test set):**

| Label | F1 Score |
|:------|:----------|
| BadEnding | 0.9274 |
| Conflict | 0.4967 |
| Dialogue | 0.9372 |
| Foreshadowing | 0.3629 |
| MoralValue | 0.8742 |
| Twist | 0.9094 |

---

## 🧠 Conclusions  

- The **pretrained DistilBERT** model outperforms the custom model in both **F1** and **accuracy**.  
- However, it is **larger (≈ 255 MB)** compared to our **custom model (≈ 87 MB)**.  
- Pretrained models simplify preprocessing (built-in tokenizer).  
- Both models exhibit **overfitting** after several epochs → early stopping recommended.  
- **Lowest-performing tags:** “Conflict” and “Foreshadowing”. These are semantically subtle and require deeper contextual understanding rather than keyword cues.

---

## 🧰 Libraries and Tools  

| Category | Libraries |
|:----------|:-----------|
| Core ML | `torch`, `transformers`, `datasets`, `numpy`, `scikit-learn` |
| Preprocessing | `re`, `tqdm`, `pickle` |
| Environment | Google Colab, Google Drive |

---

## 📚 References  
- [Hugging Face Models](https://huggingface.co/models)  
- Original TinyStories Dataset  

---

## 🧑‍💻 Authors  
**Laura Cangiotti**, **Leonardo Cittadini**, **Pavao Jancijev**




