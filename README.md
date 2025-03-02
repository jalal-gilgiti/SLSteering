**SL-Steer****
Query Plan Encoding:
SL-Steer is a query optimizer that utilizes the advanced query encoding scheme from the QueryFormer paper. We enhance the QueryFormer encoding scheme by adding explicit hint encoding and setting hints as the target.
Model: 
We use two different hybrid models for testing: one is the Transformer-BiLSTM combination, and the other is a Transformer-based regression and classification model designed for query optimization. Our main goal is to predict hints and costs based on the encoded features.








