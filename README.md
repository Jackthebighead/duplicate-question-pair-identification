

# Documentation on Duplicate Question Pair Identification Project

`Author: YUAN Yan Zhe, yanzheyuan23@sina.com`

### Description of the Project

In the domain of Natural Language Processing (NLP), text similarity is a hot spot. It is particularly important to measure the similarity between sentences or phrases in some NLP subareas such as dialog system and information retrieval. Quora Question Pairs is a Kaggle competition, which challenges participants to tackle the problem of identifying duplicate questions. This paper works on the sentence similarity problem and deals with the task of duplicate question pairs identifica- tion.

In all, we proposes three models to tackle the duplicate question pairs iden- tification problem on the Quora Question Pairs dataset. **Model 1** uses traditional machine learning classifiers like SGD Classifier, Random Forest Classifier, and XGBoost Classifier with statistical, NLP and vector features extracted from ques- tion text for classification. **Model 2** uses a Siamese BiLSTM based neural network structure combined with NLP features for classification. **Model 3** fine-tunes BERT pre-trained model and uses a simple BiLSTM based neural network structure af- ter BERT to adapt features from BERT for classification. All models achive a relatively good results.



### Description of the Files

- `sample_submission.csv`, `test.csv`, and `train.csv` are the dataset, original dataset is also accessible from Kaggle https://www.kaggle.com/c/quora-question-pairs/data.
- Model 1 consists of `feature_engineering_train.ipynb`, `feature_engineering_test.ipynb`, ``fine_tuning_process.ipynb`, `modeling.ipynb`. The first two performs feature engineering while the third one performes hyperparameter fine-tuning recording, and the last one performs modeling with traditional machine learning models.
- Model 2 consists of `lstm.ipynb` (the baseline), `lstm_featured.ipynb` (improvement), `bilstm.ipynb` (improvement), `lstm_featured-comparison.ipynb` (experiments).
- Model 3 consists of `bert.ipynb`.
- `5012proj_report` is the thesis.
- `5012proj_ppt` is the presentation material.



### Notes on people who want to reproduct this project

- I didn't upload the intermediate result (see pictures below), like the feature engineering results used in Model1. Model 2 also uses some of features (tm_feature.csv and nlp_feature.csv) from Model 1's feature engineering process. You need to run ipynb notebooks in the correct order to reproduce all my results.

  -  And also, some of my ipynb like:
     -  in the `feature_engineering_train.ipynb` i used en_core_web_md but in the `modeling.ipynb` i changed it to en_core_web_sm that i obtained from previous experiment (this is because my laptop runs out of memory if using the md version, so i give it up in the modeling process, so if you want to reproduce, use the correct version of packages and functions)
     -  in the `lstm_featured.ipynb`, you need to change to Bidirectional to get the results that i've get in the report/ppt/md_comments. 

  -  All these things need to be notices are been written in every ipynbs, read carefully. However, If you have any questions or want to get the intermediate results, feel free to reach me via `yanzheyuan23@sina.com`.

- You may not be able to run this code, use the corresponding package depending on the situation of your computer. I've write clearly the relevant requirements in each ipynb file.



### Conclusions on the Project

- This is the term project of MSBD5012 Machine Learning course offered by HKUST SENG BDT (Big Data Technology) program in my first Master semester at HKUST. My instructor is Prof. Nevin ZHANG. and big thanks to him for teaching us such a meaningful and valuable course. 

- **In all, this project explore the ways to deal with Quora Question Pairs, which is a classical problem in NLP-Text Similarity subarea.** However, many things are still not done like optimizing Model 3, trying new approaches, and model stacking, etc. Hopefully i will study more in NLP area and optimize my models in the future.



### References

**Thanks for all these references.** 

- Sanjeev Arora, Yingyu Liang, and Tengyu Ma. A simple but tough-to-beat baseline for sentence embeddings. 2016. 
- Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In *Proceedings* *of the 22nd* *acm* *sigkdd* *international* *conference* *on* *knowledge* *discovery* *and data* *mining*, pp. 785–794, 2016. 
- Zihan Chen, Hongbo Zhang, Xiaoji Zhang, and Leqi Zhao. Quora question pairs, 2018.
  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
- bidirectional transformers for language understanding. *arXiv* *preprint* *arXiv:1810.04805*, 2018. 
- Zainab Imtiaz, Muhammad Umer, Muhammad Ahmad, Saleem Ullah, Gyu Sang Choi, and Arif Mehmood. Duplicate questions pair detection using siamese malstm. *IEEE Access*, 8:21932– 21942, 2020. 
- Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. From word embeddings to document distances. In *International* *conference* *on machine* *learning*, pp. 957–966, 2015. 
- Jonas Mueller and Aditya Thyagarajan. Siamese recurrent architectures for learning sentence simi- larity. In *thirtieth* *AAAI* *conference* *on* *artificial* *intelligence*, 2016. 
- Jeffrey Pennington, R. Socher, and Christopher D. Manning. Glove: Global vectors for word repre- sentation. In *EMNLP*, 2014. 
- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert- networks. *arXiv* *preprint* *arXiv:1908.10084*, 2019. 
- Bhargav Srinivasa-Desikan. *Natural* *Language* *Processing* *and* *Computational* *Linguistics**: A* *prac**- tical guide to* *text* *analysis* *with* *Python,* *Gensim**,* *spaCy**, and* *Keras*. Packt Publishing Ltd, 2018. 
- Chi Sun, Xipeng Qiu, Yige Xu, and Xuanjing Huang. How to fine-tune bert for text classification? In *China National* *Conference* *on* *Chinese* *Computational* *Linguistics*, pp. 194–206. Springer, 2019. 
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances* *in neural information* *processing* *systems*, pp. 5998–6008, 2017. 
- Lingxi Zhang, Zhiyong Feng, Wei Ren, and Hong Luo. Siamese-based bilstm network for scratch source code similarity measuring. In *2020 International Wireless Communications and Mobile* *Computing* *(IWCMC)*, pp. 1800–1805. IEEE, 2020. 
- Sanjeev Arora, Yingyu Liang, and Tengyu Ma. A simple but tough-to-beat baseline for sentence embeddings. 2016. 
- Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In *Proceedings* *of the 22nd* *acm* *sigkdd* *international* *conference* *on* *knowledge* *discovery* *and data* *mining*, pp. 785–794, 2016. 
- Zihan Chen, Hongbo Zhang, Xiaoji Zhang, and Leqi Zhao. Quora question pairs, 2018.
  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
- bidirectional transformers for language understanding. *arXiv* *preprint* *arXiv:1810.04805*, 2018. 
- Zainab Imtiaz, Muhammad Umer, Muhammad Ahmad, Saleem Ullah, Gyu Sang Choi, and Arif Mehmood. Duplicate questions pair detection using siamese malstm. *IEEE Access*, 8:21932– 21942, 2020. 
- Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. From word embeddings to document distances. In *International* *conference* *on machine* *learning*, pp. 957–966, 2015. 
- Jonas Mueller and Aditya Thyagarajan. Siamese recurrent architectures for learning sentence simi- larity. In *thirtieth* *AAAI* *conference* *on* *artificial* *intelligence*, 2016. 
- Jeffrey Pennington, R. Socher, and Christopher D. Manning. Glove: Global vectors for word repre- sentation. In *EMNLP*, 2014. 
- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert- networks. *arXiv* *preprint* *arXiv:1908.10084*, 2019. 
- Bhargav Srinivasa-Desikan. *Natural* *Language* *Processing* *and* *Computational* *Linguistics**: A* *prac**- tical guide to* *text* *analysis* *with* *Python,* *Gensim**,* *spaCy**, and* *Keras*. Packt Publishing Ltd, 2018. 
- Chi Sun, Xipeng Qiu, Yige Xu, and Xuanjing Huang. How to fine-tune bert for text classification? In *China National* *Conference* *on* *Chinese* *Computational* *Linguistics*, pp. 194–206. Springer, 2019. 
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances* *in neural information* *processing* *systems*, pp. 5998–6008, 2017. 
- Lingxi Zhang, Zhiyong Feng, Wei Ren, and Hong Luo. Siamese-based bilstm network for scratch source code similarity measuring. In *2020 International Wireless Communications and Mobile* *Computing* *(IWCMC)*, pp. 1800–1805. IEEE, 2020. 
- https://blog.csdn.net/u010657489/article/details/51952785
- [https://github.com/seatgeek/fuzzywuzzy#usage](https://github.com/seatgeek/fuzzywuzzy)
- [https://www.kaggle.com/amoyyean/lstm-with-glove](http://static.hongbozhang.me/doc/STAT_441_Report.pdf )
- https://huggingface.co/transformers/model_doc/bert.html
- [https://keras.io/api/layers/recurrent_layers/bidirectional/](https://huggingface.co/transformers/model_doc/bert.html)
- [https://keras.io/api/layers/recurrent_layers/lstm/](https://huggingface.co/transformers/model_doc/bert.html)
- https://keras.io/examples/nlp/semantic_similarity_with_bert/
- https://keras.io/examples/nlp/masked_language_modeling/
- https://www.sbert.net/docs/training/overview.html?highlight=get_word_embedding_dimension
- [https://github.com/hanxiao/bert-as-service#building-a-qa-semantic-search-engine-in-3-minutes](https://github.com/hanxiao/bert-as-service)
- https://github.com/dongfang91/text_similarity
- https://github.com/dimichai/quora-question-pairs/blob/master/bilstm_features.ipynb
- [https://github.com/qqgeogor/kaggle-quora-solution-8th/tree/master/model_gzf/stack](https://spacy.io/models/en)
- https://github.com/TanyaChutani/Siamese-Network-On-Text-Data
- https://spacy.io/models/en
- http://www.nltk.org/
- https://radimrehurek.com/gensim/


