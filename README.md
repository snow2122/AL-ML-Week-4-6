Week 4: News Topic Classifier Using BERT

For Task 1, I worked on fine-tuning a BERT model for text classification using the AG News dataset. I started by preprocessing the dataset with a tokenizer to convert text into numerical form suitable for the model. After setting up the training and evaluation splits, I defined accuracy and F1 score as evaluation metrics to assess performance. To make training feasible on limited hardware (CPU), I reduced batch sizes and epochs while still maintaining meaningful learning. The model was then trained on the dataset and evaluated for its effectiveness in predicting news categories. This task helped me understand the complete flow of preparing data, training a transformer-based model, and evaluating its performance in a resource-constrained environment.

Week 5: End-to-End ML Pipeline with Scikit-learn Pipeline API
For Task 2, I developed an end-to-end machine learning pipeline to predict customer churn using the Telco Churn dataset. The focus was on creating a production-ready workflow by automating preprocessing steps such as encoding categorical features and scaling numerical ones. I implemented Logistic Regression and Random Forest models within the pipeline and applied GridSearchCV for hyperparameter tuning to optimize performance. The pipeline was then exported with joblib for reusability, ensuring that the entire preprocessing and modeling process could be easily replicated in production. This task strengthened my understanding of pipeline design, hyperparameter tuning, and building reproducible ML workflows.

Week 6: Auto Tagging Support Tickets Using LLM

For Task 5, I worked on using large language models (LLMs) to automatically categorize support tickets into predefined tags. The task explored multiple approaches: zero-shot classification using direct prompts, few-shot learning by providing examples, and the idea of fine-tuning for more accurate results. The model was designed to output the top three most probable tags for each ticket, which provides flexibility for real-world applications where multiple categories may overlap. This exercise helped me understand how prompt engineering, zero-shot learning, and ranking predictions can be applied to practical NLP problems. It also highlighted the trade-off between ease of use with pre-trained LLMs and the improved accuracy that fine-tuning can bring.




