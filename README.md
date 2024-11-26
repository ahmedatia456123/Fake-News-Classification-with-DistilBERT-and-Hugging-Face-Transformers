# Fake News Detection Using DistilBERT, MobileBERT, and TinyBERT

## Overview
This repository demonstrates a complete workflow for fake news detection using state-of-the-art transformer-based models: **DistilBERT**, **MobileBERT**, and **TinyBERT**. The notebook highlights the use of Hugging Face's `transformers` library to tokenize, fine-tune, and evaluate these models on a custom dataset. The project is structured to showcase a **skilled machine learning workflow** with clear explanations and impressive results.

---

## 📂 Project Workflow
1. **Data Preprocessing:**
   - Loaded and cleaned the dataset (`fake_news.xlsx`).
   - Visualized label distribution and text token characteristics.
2. **Dataset Splitting:**
   - Created train, validation, and test splits while preserving label distribution.
   - Converted DataFrames into `Hugging Face DatasetDict` for streamlined processing.
3. **Tokenization:**
   - Compared tokenization with **DistilBERT**, **MobileBERT**, and **TinyBERT**.
   - Utilized `Hugging Face AutoTokenizer` for fast and efficient text tokenization.
4. **Model Training:**
   - Configured and fine-tuned the **DistilBERT** model with custom training arguments.
   - Used the Hugging Face `Trainer` API for streamlined training and evaluation.
5. **Evaluation:**
   - Achieved an impressive **92.2% accuracy** on the test set.
   - Detailed classification metrics and performance breakdown are provided.

---

## 🚀 Results

### Training and Validation Performance
| Epoch | Validation Loss | Accuracy  |
|-------|-----------------|-----------|
| 1     | 0.175188        | 93.7%     |
| 2     | 0.165203        | 93.2%     |
| 3     | 0.169197        | 93.4%     |

### Test Set Metrics
- **Test Accuracy:** 92.2%
- **Precision, Recall, F1-Score:**
  | Class       | Precision | Recall | F1-Score | Support |
  |-------------|-----------|--------|----------|---------|
  | Reliable    | 93%       | 93%    | 93%      | 404     |
  | Unreliable  | 91%       | 91%    | 91%      | 327     |
  | **Overall** | **92%**   | **92%**| **92%**  | **731** |

---

## 🔧 Key Features
- **Efficient Data Handling:** Optimized preprocessing and tokenization for large datasets.
- **Transformer Models:** Demonstrates flexibility by integrating DistilBERT, MobileBERT, and TinyBERT.
- **Performance Visualizations:** Histograms for word and token distributions provide deeper insights into the dataset.
- **Metrics-Driven Evaluation:** Comprehensive use of metrics like accuracy, precision, recall, and F1-score.

---

## 📊 Visualizations
### Label Distribution
![Label Distribution](label_distribution.png)

### Token Distribution
| Metric             | Average | Max   | Min   |
|--------------------|---------|-------|-------|
| Title Tokens       | 15.3    | 30    | 5     |
| Text Tokens        | 112.7   | 300   | 15    |

![Token Histogram](token_histograms.png)

---

## 🛠️ Libraries and Tools
- **Transformers:** Hugging Face's library for tokenization and model fine-tuning.
- **Pandas, NumPy:** For data manipulation and processing.
- **Matplotlib, Seaborn:** For data visualization.
- **Scikit-learn:** For evaluation metrics.
- **UMAP:** Optional for dimensionality reduction.

---

## 🤔 Why This Project Stands Out
- **Job-Ready Skills:** Demonstrates a clear understanding of transformer-based NLP models, preprocessing pipelines, and evaluation metrics.
- **Well-Documented Process:** Every step is explained with insights into **why** it's done and **how** it impacts the outcome.
- **Results-Oriented Approach:** Focused on producing tangible metrics and insights to validate the model's performance.

---

## 📈 Conclusion
This project exemplifies the practical use of advanced transformer models for a real-world NLP problem. By leveraging pre-trained models and customizing workflows, the results demonstrate my ability to deliver high-quality machine-learning solutions.

---

## 🤝 Connect with Me
- **Email:** [ahmedatia456123@gmail.com](mailto:ahmedatia456123@gmail.com)
- **GitHub:** [GitHub Profile](https://github.com/ahmedatia456123)
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/ahmedatia456123)
