# RecommenderSystem
This is an anime and manga recommender system based off of MyAnimeList, AniList, Kitsu, and Anime-Planet user ratings. This codebase has been used to train a model on over 1.5 billion user-item interactions. See the notebook in `notebooks/README.ipynb` for usage instructions.

Details on the recommender system can be found by inspecting the source code at `notebooks/TrainingAlphas`. At a high level, there are four main steps:
1. Pretraining a transformer model and a bag-of-items model on watch histories
2. Finetuning the models on recent data to predict the following metrics:
   * The probability that you will watch a series
   * The rating that you will give to the series
   * The probability that you will finish a series after starting it
   * The probability that you will add the series to your plan-to-watch list
3. Using a ranking model to combine these metrics into a relevance score
4. Reranking the outputs to generate a list of recommendations

See [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf), [Position-Aware ListMLE: A Sequential Learning Process for Ranking](https://auai.org/uai2014/proceedings/individuals/164.pdf), and [Deep Learning for Recommender Systems: A Netflix Case Study](https://ojs.aaai.org/index.php/aimagazine/article/view/18140) for relevant prior work.
