# RecommenderSystem
This is a recommender system for anime and manga that is trained on over 1.5 billion user-item interactions from MyAnimeList, AniList, Kitsu, and Anime-Planet. See `notebooks/README.ipynb` for usage instructions.

Details on the recommender system can be found by inspecting the source code at `notebooks`. The main steps are
1. Stitching multiple snapshots of a user's list to create a timestamped history of interactions
2. Pretraining a llama3 style transformer on the watch histories
3. Finetuning the model on recent data to predict, for each (user, item) pair:
   * the probability of positive engagement
   * the user's rating for the item
   * the user's watching status

Once trained, the models are containerized and deployed on gpu instances. A website, which is currently in private beta, queries the endpoint and lets users view their recommendations. More information about the website can be found in the source code at `notebooks/Package/Client`. The models are continuously updated to incorporate new items as they release and to reduce temporal drift.

See [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152), [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf), and [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) for relevant prior work.
