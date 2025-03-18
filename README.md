**SL-Steer: Generalizable and Cost-Aware Query Optimization for SQL Workloads
**

**ABSTRACT**

Query optimization is a critical challenge in database management
systems (DBMSs), where execution efficiency is essential for han-
dling complex workloads. Traditional optimizers, which rely on
heuristic-driven cost models, often deliver suboptimal performance
due to their limited ability to generalize across diverse queries.
Steering-based optimization, which utilizes database-exposed hints,
presents a promising alternative. However, current methods face
several significant drawbacks: (1) They generate query plans by
iteratively approximating the plan space, reducing generalizability
and adaptability while increasing optimization overhead, especially
for complex or unseen queries. (2) They rely on predefined hint sets
and greedy search strategies, resulting in higher query execution
latency. (3) They predict execution costs for all generated plans and
use computationally expensive contextual bandits to select the op-
timal plan, further increasing delays. (4) Existing approaches fail to
fully leverage rich query features for encoding, limiting both their
optimization accuracy and efficiency. To address these challenges,
we introduce Snow Leopard Steer (SL-Steer), a novel deep learn-
ing framework that leverages the entire pool of query plans while
dynamically prioritizing the most efficient ones. By employing the
Dynamic Cost Aware Prioritization (DCAP) approach, SL-Steer en-
ables full query space training while focusing on low execution
time queries, enhancing generalization for unseen queries. Unlike
existing methods, SL-Steer directly predicts the optimal hint set
and execution cost in a single step, eliminating the need for plan
approximation and avoiding greedy searches and computationally
expensive sorting mechanisms. The framework utilizes a hybrid
Transformer-BiLSTM model OptiFusion (see subsection 3.3) with
multi-head attention to effectively extract the most relevant query
plan features, ensuring both accuracy and efficiency in optimiza-
tion. Extensive evaluations demonstrate that SL-Steer achieves up
to 3× faster query optimization, significantly reducing latency by 5%
to 10% and improving accuracy. It outperforms heuristic-based ap-
proaches, providing a scalable and adaptable solution across various
database engines.

**Query Plan Encoding:**
SL-Steer is a query optimizer that utilizes the advanced query encoding scheme from the QueryFormer paper. We enhance the QueryFormer encoding scheme by adding explicit hint encoding and setting hints as the target.

**Model:**
We use two different hybrid models for testing: one is the Transformer-BiLSTM combination, and the other is a Transformer-based regression and classification model designed for query optimization. Our main goal is to predict hints and costs based on the encoded features.








