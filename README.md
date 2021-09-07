This repository contains model solutions for duplicate question(sentence) pair classification on QQP dataset, which solves the problem of Semantic Text Simmilarity in NLP. This is a continuous updating repository. Newly updates are shown before the documentation of the projects. **Some codes after 2020.12.21 are not updated now in this repo, and they will be released soon**.

### Newly update (2021.1.15)
- implemented [Enhanced RCNN](https://dl.acm.org/doi/pdf/10.1145/3366423.3379998).
- model stacking, tbc..

### Update (2020.12.21)
- Model results in the report (for 5012:ML course project only) will remain the status on DEC 3rd, 2020.
- sentence BERT(Siamese BERT) is tried. 2020.12.21 [sbert](https://arxiv.org/pdf/1908.10084.pdf)
- ESIM is tried. 2020.12.21 [esim](https://arxiv.org/pdf/1609.06038.pdf)


# Duplicate Question Pair Identification Project

### Description of the Project

In the domain of Natural Language Processing (NLP), text similarity is a hot spot. It is particularly important to measure the similarity between sentences or phrases in some NLP subareas such as dialog system and information retrieval. Quora Question Pairs is a Kaggle competition, which challenges participants to tackle the problem of identifying duplicate questions. This paper works on the sentence similarity problem and deals with the task of duplicate question pairs identification.

In all (so far as for the ML course project), we proposes three models to tackle the duplicate question pairs identification problem on the Quora Question Pairs dataset. **Model 1** uses traditional machine learning classifiers like SGD Classifier, Random Forest Classifier, and XGBoost Classifier with statistical, NLP and vector features extracted from ques- tion text for classification. **Model 2** uses a Siamese BiLSTM based neural network structure combined with NLP features for classification. **Model 3** fine-tunes BERT pre-trained model and uses a simple BiLSTM based neural network structure after BERT to adapt features from BERT for classification. All models achive a relatively good results.



### Introduction of the Files

-  `sample_submission.csv`, `test.csv`, and `train.csv` are the dataset, original dataset is also accessible from Kaggle https://www.kaggle.com/c/quora-question-pairs/data.
- Model 1 consists of `feature_engineering_train.ipynb`, `feature_engineering_test.ipynb`, ``fine_tuning_process.ipynb`, `modeling.ipynb`. The first two performs feature engineering while the third one performes hyperparameter fine-tuning recording, and the last one performs modeling with traditional machine learning models.
- Model 2 consists of `siamese_lstm.ipynb` (the baseline), `siamese_lstm_featured.ipynb` (improvement), `siamese_bilstm.ipynb` (improvement), `siamese_lstm_featured-comparison.ipynb` (experiments).
- Model 3 consists of `bert.ipynb`.
- `Duplicate_Question_Pairs_Identification.pdf` is the thesis.
- `link.txt` contains the video link presenting our work via YouTube.



### Notes on Reproduction

- I didn't upload the intermediate result (see pictures below), like the feature engineering results used in Model1. Model 2 also uses some of features (tm_feature.csv and nlp_feature.csv) from Model 1's feature engineering process. You need to run ipynb notebooks in the correct order to reproduce all my results.

  -  And also, some of my ipynb like:
    -  in the `feature_engineering_train.ipynb` i used en_core_web_md but in the `modeling.ipynb` i changed it to en_core_web_sm that i obtained from previous experiment (this is because my laptop runs out of memory if using the md version, so i give it up in the modeling process, so if you want to reproduce, use the correct version of packages and functions)
    - in the `siamese_lstm_featured.ipynb`, you need to change to Bidirectional to get the results that i've get in the report/ppt/md_comments. 

  -  All these things need to be notices are been written in every ipynbs, read carefully. However, If you have any questions or want to get the intermediate results, feel free to reach me via **`yanzheyuan23@sina.com`**.

- You may not be able to run this code, use the corresponding package depending on the situation of your computer. I've write clearly the relevant requirements in each ipynb file.

- There are other mehtods that I tried but didn't put into this repo and the course project report: Fine-tuned BERT(compared to model 3), sentence BERT(Siamese BERT), ESIM with Features (compared with model 2). If you have any need, feel free to contact me. 
  - (In all, this project explore the ways to deal with Quora Question Pairs, which is a classical problem in NLP-Text Similarity subarea. However, many things are still not done like optimizing Model 3, trying new approaches, and model stacking, etc. Hopefully i will study more in NLP area and optimize my models in the future.)



### Acknowledgements

- This is a ML project under the supervision of Prof. Nevin ZHANG. and big thanks to him for helping me.



### References

**Thanks for all these references.** 

- Sanjeev Arora, Yingyu Liang, and Tengyu Ma. A simple but tough-to-beat baseline for sentence embeddings. 2016. 
  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
- bidirectional transformers for language understanding. *arXiv* *preprint* *arXiv:1810.04805*, 2018. 
- Zainab Imtiaz, Muhammad Umer, Muhammad Ahmad, Saleem Ullah, Gyu Sang Choi, and Arif Mehmood. Duplicate questions pair detection using siamese malstm. *IEEE Access*, 8:21932– 21942, 2020. 
- Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. From word embeddings to document distances. In *International* *conference* *on machine* *learning*, pp. 957–966, 2015. 
- Jonas Mueller and Aditya Thyagarajan. Siamese recurrent architectures for learning sentence simi- larity. In *thirtieth* *AAAI* *conference* *on* *artificial* *intelligence*, 2016..
- Bhargav Srinivasa-Desikan. *Natural* *Language* *Processing* *and* *Computational* *Linguistics**: A* *prac**- tical guide to* *text* *analysis* *with* *Python,* *Gensim**,* *spaCy**, and* *Keras*. Packt Publishing Ltd, 2018. 
- Chi Sun, Xipeng Qiu, Yige Xu, and Xuanjing Huang. How to fine-tune bert for text classification? In *China National* *Conference* *on* *Chinese* *Computational* *Linguistics*, pp. 194–206. Springer, 2019. 


### Liscense
Please refer to: [liscense_Jackthebighead](https://github.com/Jackthebighead/Duplicate-Question-Pairs-Identification/blob/main/LICENSE)
