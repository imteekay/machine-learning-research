# Deep Learning for Biology

## Success Criteria

**Performance metric (e.g., accuracy, AUC, F1)**: You might aim to match the performance of a human expert, achieve a correlation with experimental results comparable to a technical replicate, or keep the false-positive rate below a certain number.

**Level of interpretability**: In many applications, it’s important not only that a model performs well, but also that its decisions can be understood by domain experts. For instance, you may prioritize well-calibrated uncertainty estimates or interpretable feature attributions, especially when trust and explainability are critical.

**Model size or inference latency**: If your model needs to operate in a resource-constrained environment (e.g., smartphones or embedded devices) or meet real-time throughput targets (e.g., process 20 frames per second), your success criterion might focus on efficiency—such as achieving high performance per floating point operation (FLOP), which measures how effectively the model uses computational resources. In such cases, metrics like inference time, memory usage, or energy consumption may matter more than raw accuracy.

**Training time and efficiency**: When compute is limited—or for educational contexts—you may prioritize fast training or minimal hardware requirements. Since training deep learning models typically involves large matrix operations, they are often accelerated using graphics processing units (GPUs). In low-resource settings, developing a simpler model that trains quickly on a CPU may be a more practical goal than maximizing performance.

**Generalizability**: In some cases, the goal is to build a model that works well across many datasets or tasks, rather than one that is finely tuned to a single benchmark. For example, foundational models—large models trained on broad datasets that can be adapted to many downstream applications—prioritize flexibility and reuse. In such settings, broad applicability may be more valuable than squeezing out the best possible performance on a specific task.

## Invest Heavily in Evaluations

Thinking carefully about precisely how you’ll measure progress—including what metrics you’ll use, how you’ll validate results, and which baselines you’ll compare against. Without a clear, well-designed evaluation strategy, even a technically impressive model can fail to produce meaningful conclusions.

## Designing Baselines

### Classification tasks

**Random prediction**: Assign labels completely at random, with equal probability for each class. This tells you what performance looks like with no information at all.

**Random prediction weighted by class frequencies**: Sample labels randomly, but in proportion to how often they occur in the training data. This is useful for imbalanced datasets.

**Majority class**: Always predict the most common class. This can be a surprisingly hard baseline to beat in highly class imbalanced settings.

**Nearest neighbor**: Predict the label of the most similar example in the training data (e.g., 1-nearest neighbor using Euclidean distance). This is often effective when inputs are low dimensional or well structured.

### Regression tasks

**Mean or median of the target**: Always predict the average or median target value from the training set. This often matches what a model would do if it’s not learning anything meaningful.

**Linear regression with a single feature**: Fit a line using just the strongest individual predictor (e.g., one biomarker). This helps gauge how much a more complex model improves over a simple signal.

**K-nearest neighbor regression**: Predict the target as the average (or weighted average) of the k most similar data points. This is simple to implement and often surprisingly competitive on structured datasets.

### For both

**Simple heuristics**: Use straightforward rules based on domain knowledge. For example, in diagnostics, classify a patient as positive if a single biomarker or measurement exceeds a threshold. For skin cancer images, rank lesions by average pixel intensity. In genomics, if the task is to predict which gene a mutation affects, a simple baseline is to assume it affects the nearest gene in the genome.
