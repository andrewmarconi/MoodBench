"""
Methodology UI components for MoodBench Gradio interface.
"""


def create_methodology_tab():
    """Create the methodology tab for the Gradio interface."""
    import gradio as gr

    with gr.TabItem("📚 Methodology"):
        gr.Markdown("""
# Methodology & Data Documentation

## 📊 Data Provenance

### Disney Customer Reviews Dataset
- **Source**: [Kaggle - Disneyland Reviews](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews)
- **Size**: 42,000+ customer reviews
- **Locations**:
  - Disneyland California (USA)
  - Disneyland Paris (France)
  - Hong Kong Disneyland
- **Rating Scale**: 1-5 stars
- **Content**: Customer-written text reviews with associated ratings

### Training Datasets
Models are fine-tuned on four standard sentiment analysis benchmarks:
- **IMDB**: [Stanford Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) - 50k movie reviews
- **SST-2**: [Stanford Sentiment Treebank](https://huggingface.co/datasets/stanfordnlp/sst2) - Single sentence reviews
- **Amazon**: [Amazon Polarity](https://huggingface.co/datasets/amazon_polarity) - Product reviews
- **Yelp**: [Yelp Polarity](https://huggingface.co/datasets/yelp_polarity) - Business reviews

All datasets are publicly available and accessed via Hugging Face Datasets or Kaggle.

---

## 🧮 Calculation Methodology

### Actual NPS from Customer Ratings
**Actual NPS** gauges show the true customer sentiment based on their star ratings:

**Rating Mapping:**
```
5 stars → Promoter    (9-10 on NPS scale)
4 stars → Passive     (7-8 on NPS scale)
1-3 stars → Detractor (0-6 on NPS scale)
```

**Calculation:**
```python
NPS Score = (% Promoters) - (% Detractors)
```

**Example:**
- If 50% gave 5★, 30% gave 4★, and 20% gave 1-3★
- NPS = 50% - 20% = **+30**

**Range**: -100 (all detractors) to +100 (all promoters)

### Estimated NPS from Model Predictions
**Estimated NPS** charts show how models predict sentiment:

**Model Output Mapping:**
- Models predict **binary sentiment** (positive/negative) with **confidence scores**
- High-confidence positive (>0.85) → Promoter
- Medium-confidence positive (0.60-0.85) → Passive
- Negative predictions → Detractor

**Purpose**: Compare model predictions against actual customer sentiment to assess model quality.

### Model Accuracy by Training Dataset
**Accuracy** measures how well models predict the correct sentiment (positive/negative):

**Calculation:**
```python
Accuracy = (Correct Predictions / Total Predictions) × 100%
```

**Charts show:**
- X-axis: Training dataset (IMDB, SST-2, Amazon, Yelp)
- Y-axis: Accuracy percentage
- Red dashed line: Average accuracy across all training datasets

**Interpretation**: Higher accuracy = better transfer learning from training dataset to Disney reviews.

### NPS Categories by Model
**Stacked bar charts** show the distribution of predictions:
- **Green (Promoters)**: Models predict high customer satisfaction
- **Yellow (Passives)**: Models predict moderate satisfaction
- **Red (Detractors)**: Models predict dissatisfaction

**Aggregation**: Counts are summed across all training dataset variants of each base model, then normalized to 100%.

---

## ⚠️ Caveats & Limitations

### Data Quality
- **Sample Bias**: Kaggle dataset may not represent all Disney customers
- **Self-Selection**: Online reviewers may have stronger opinions than average visitors
- **Time Period**: Reviews reflect historical sentiment, not current conditions
- **Language**: Analysis assumes English-language reviews only

### Model Limitations
- **Binary Sentiment**: Models only predict positive/negative, missing nuanced emotions
- **Context Window**: Longer reviews may be truncated (max 512 tokens)
- **Training Domain Mismatch**: Models trained on movies/products may not fully understand theme park experiences
- **Confidence Calibration**: Confidence thresholds (0.85, 0.60) are heuristic, not rigorously calibrated

### NPS Methodology
- **Non-Standard Mapping**: Traditional NPS uses 0-10 scale; we map 1-5 stars with assumptions
- **Missing Neutrality**: 4-star reviews (Passives) may include both satisfied and slightly dissatisfied customers
- **Aggregation Effects**: Combining different training datasets may introduce noise

### Cross-Dataset Evaluation
- **Domain Shift**: Theme park reviews differ significantly from movie/product reviews
- **Generalization**: High accuracy on training data doesn't guarantee Disney review accuracy
- **Class Imbalance**: Disney reviews may skew more positive than training datasets

### Technical Constraints
- **Model Size**: Small models (4M-410M parameters) have inherent performance limits
- **Fine-Tuning**: LoRA adapters may not capture full semantic complexity
- **Quantization**: 4-bit quantization trades accuracy for memory efficiency

---

## 🎯 Recommended Interpretation

### When to Trust the Data
- **Relative Comparisons**: Comparing models or locations within this dataset
- **Trend Identification**: Identifying which training datasets transfer better
- **Model Selection**: Choosing which model architecture works best for this domain

### When to Be Cautious
- **Absolute NPS Values**: Don't compare these NPS scores to industry benchmarks
- **Real-World Decisions**: Don't base business decisions solely on these estimates
- **Causation Claims**: Correlation between training data and accuracy doesn't imply causation
- **Generalization**: Results may not apply to other theme parks or time periods

### Best Practices
1. **Use as Proxy**: Treat estimated NPS as a proxy for model quality, not ground truth
2. **Compare Within Dataset**: Only compare metrics within this controlled experiment
3. **Validate Externally**: Cross-reference findings with official Disney satisfaction data if available
4. **Consider Ensemble**: Average predictions from multiple models for more robust estimates
5. **Monitor Drift**: Re-evaluate if applying to significantly newer review data

---

## 📖 Additional Resources

- **Hugging Face Transformers**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **LoRA Fine-Tuning**: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- **Net Promoter Score**: [https://en.wikipedia.org/wiki/Net_promoter_score](https://en.wikipedia.org/wiki/Net_promoter_score)
- **Sentiment Analysis Datasets**: [https://paperswithcode.com/task/sentiment-analysis](https://paperswithcode.com/task/sentiment-analysis)

---

*This documentation was generated as part of the MoodBench project to provide transparency about data sources, calculation methods, and appropriate interpretation of results.*
        """)
