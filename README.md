# English-to-Hindi Machine Translation Using Transformer Models

## 1. Project Overview

This project explores English-to-Hindi machine translation using Transformer-based neural networks.

The objective is to implement and compare encoder–decoder and decoder-only architectures, evaluate their translation quality, and examine their computational requirements and responsible AI considerations.

The project uses the IIT Bombay English-Hindi Parallel Corpus.

## 2. Dataset Description

**Dataset:** IIT Bombay English-Hindi Parallel Corpus

**Source:** https://huggingface.co/datasets/cfilt/iitb-english-hindi

The dataset contains parallel English and Hindi sentences used for supervised machine translation.

The project uses a manageable subset of the corpus because of available computational resources.

### Preprocessing

- Remove empty English or Hindi sentences.
- Remove excessively long sentence pairs.
- Shuffle the dataset using a fixed random seed.
- Tokenize the English and Hindi sentences using the tokenizer associated with each model.
- Divide the dataset into training, validation, and test sets.

The main custom Transformer and mT5 experiments use a split of 4,732 training examples, 591 validation examples, and 592 test examples.

An earlier MarianMT baseline experiment used a separate 54,583-example cleaned subset. Its results should not be directly compared with the other experiments unless they are evaluated again on the same test set.

## 3. Models

### 3.1 Basic Transformer

A custom encoder–decoder Transformer was implemented using PyTorch and trained from random initialization.

The architecture contains:
- Token embeddings
- Positional embeddings
- Transformer encoder layers
- Transformer decoder layers
- Attention masks
- Output projection layer

The custom implementation was trained for three epochs.

### 3.2 mT5-small

The pretrained multilingual mT5-small model was adapted for English-to-Hindi translation using supervised fine-tuning.

The model uses an encoder–decoder architecture and the instruction prefix:

`translate English to Hindi:`

The current notebook used a 300-example training subset during CPU execution.

### 3.3 LLaMA

Planned model: Meta Llama 3.2 1B Instruct.

Architecture: Decoder-only Transformer.

The model will be adapted for English-to-Hindi translation using an appropriate prompting or parameter-efficient fine-tuning approach.

### 3.4 Gemma

Planned model: Gemma 2 2B IT.

Architecture: Decoder-only Transformer.

The model will be adapted for English-to-Hindi translation using prompting or parameter-efficient fine-tuning.

## 4. Implementation

The experiments are implemented in Google Colab using Python and PyTorch.

The main notebook is:

`English_Hindi_Transformer.ipynb`

### Main dependencies

- Python
- PyTorch
- Transformers
- Hugging Face Datasets
- SentencePiece
- SacreBLEU
- Accelerate
- Pandas
- NumPy

## 5. Evaluation Metrics

The project evaluates translation systems using:

- **BLEU:** Measures n-gram overlap between generated and reference translations.
- **chrF:** Measures character n-gram similarity.
- **Training/Fine-tuning time:** Measures the time required to adapt or train a model.
- **Inference time:** Measures the time required to generate translations.
- **Model parameters:** Reports the total model size and, where applicable, trainable parameters.

A final comparison must use the same test examples and consistent generation settings for all four models.

## 6. Preliminary Experimental Results

### Basic Transformer trained from scratch

| Metric | Result |
|---|---:|
| Training examples | 4,732 |
| Validation examples | 591 |
| Test examples | 592 |
| Evaluated examples | 200 |
| Epochs | 3 |
| BLEU | 0.0129 |
| chrF | 1.9285 |
| Training time | 2,496.45 seconds |
| Average inference time | 1.1489 seconds |
| Trainable parameters | 8,863,616 |

### mT5-small

| Metric | Result |
|---|---:|
| Total parameters | 300,176,768 |
| Training examples used in CPU run | 300 |
| Fine-tuning time | 1,886.3 seconds |
| BLEU | Pending evaluation |
| chrF | Pending evaluation |
| Average inference time | Pending evaluation |

The final LLaMA, Gemma, and common-test-set comparison results will be added after their evaluation experiments are completed.

## 7. Sampling Temperature Analysis

The effect of generation temperature will be examined using values of 0.2, 0.7, and 1.0.

The analysis will consider:
- Translation consistency
- Fluency
- Semantic accuracy
- Variation in generated translations
- BLEU and chrF differences

For the primary model comparison, generation settings will be kept consistent. Temperature experiments will be reported separately.

## 8. Responsible AI and Ethical Considerations

### Bias
Machine translation models can reproduce gender, social, and cultural biases present in their training data.

### Hallucination
Generative models may add information that is not present in the original English sentence.

### Cultural misrepresentation
Idioms, regional expressions, names, and culturally specific references may be translated incorrectly.

### Sensitive content
Incorrect translation of medical, legal, political, or personal information may cause harm.

### Mitigation measures
- Human review for important translations.
- Testing across different topics and contexts.
- Preserving names, numbers, and sensitive terminology.
- Avoiding the use of unverified translations for medical or legal decisions.
- Protecting sensitive input data.
- Documenting model limitations.

## 9. Conclusion

The project investigates the trade-offs between training a Transformer from scratch and adapting pretrained multilingual and generative language models.

The final conclusion will identify the best-performing model using measured translation quality, inference time, parameter count, and computational requirements.

## 10. References

1. IIT Bombay English-Hindi Parallel Corpus: https://huggingface.co/datasets/cfilt/iitb-english-hindi
2. mT5-small: https://huggingface.co/google/mt5-small
3. LLaMA 3.2 1B Instruct: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
4. Gemma 2 2B IT: https://huggingface.co/google/gemma-2-2b-it
5. Hugging Face Transformers: https://huggingface.co/docs/transformers
