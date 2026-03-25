# Multiplex Graph Prompt Learning and Attentive Fusion for Event Graph Completion

This is the code of the paper _Multiplex Graph Prompt Learning and Attentive Fusion for Event Graph
Completion_. Accepted by Neural Networks.

We propose an Multiplex Graph **P**rompt **L**earning and **A**ttentive **F**usion for Event Graph Completion (**PLAF**) model for the EGC task to learn event representations interactively among each homogeneous graphs and adopt a kind of dual graph prompt learning for missing relation prediction.

The PLAF model includes three key modules: (1) Dual Graph Prompt Learning (DGPL) reformulates each EG into an event triplet sequence to encode its structural and semantic information; (2) Multiplex Graph  Attention Network (MGAT) divides each heterogeneous EG into four homogeneous graphs to learn events’ representations through both inter-graph and cross-graph attention; (3) Aggregative Relation Prediction Module (ARPM) aggregates the missing relation predictions from both DGPL module and MGAT module to complete the EGs. Furthermore, we have constructed the EGC-MAVEN dataset and conducted extensive experiments to evaluate the efficacy of our approach. The experimental results validate our arguments and demonstrate that our proposed PLAF model outperforms the advanced competitors.
