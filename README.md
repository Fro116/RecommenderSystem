# Anime Recommendations and Manga Recommendations
This is a recommender system for anime and manga that is trained on over 1.8 billion user-item interactions from MyAnimeList, AniList, Kitsu, and Anime-Planet.

Details on the recommender system can be found by inspecting the source code at `notebooks`. The main steps are
1. Stitching multiple snapshots of a user's list to create a timestamped history of interactions.
2. Training a rating model to predict the score that the user will give to an item. We follow an approach similar to [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152).
3. Training a retrieval model to predict the next item a user will watch. We use a cloze objective similar to [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf), but with modern transformer blocks and training recipes.
4. Training a similarity model to suggest items that are semantically similar to a reference anime or manga. We take inspiration from [LambdaRank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
5. Finetuning the models daily on recent data.

Once trained, the models are containerized and deployed on gpu instances. A website, which is currently in private beta and is pending release, queries this endpoint and lets users view their recommendations.
