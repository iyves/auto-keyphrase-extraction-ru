from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from helpers import get_file_names, get_text
from IPython import display
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.ERROR)
pd.set_option('display.max_colwidth', None)


class TopicModeler(object):
    """Performs topic modeling on a corpus. The corpus should be in the form of a list of
    document tokens. Topic modeling is performed through gensim's implementation of LDA.
    """

    @classmethod
    def load_corpus(cls, year_range: List[int], input_dir: Path) -> List[List[str]]:
        input_dir = str(input_dir)
        corpus = []
        for year in year_range:
            input_file = Path(input_dir, str(year) + ".pickle")
            logging.info("Loading preprocessed text from " + str(input_file))
            with open(input_file, "rb") as file:
                corpus.extend(pickle.load(file))
        return corpus

    @classmethod
    def create_dictionary(cls, document_tokens: List[List[str]],
                          no_below: int = 5, no_above: float = 0.8):
        """Create a dictionary from the corpus of document tokens.

        Args:
            document_tokens:
                The corpus as a list of document tokens, where each document is a
                list of candidate keyphrase tokens.
            no_below:
                The minimum amount of documents in which a token must appear
                to be considered
            no_above:
                The maximum % of documents in which a token may appear in to be considered

        Returns:
            A gensim dictionary that is a mapping of a unique id to each token in the
            corpus.
        """
        dictionary = corpora.Dictionary(document_tokens)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        return dictionary

    @classmethod
    def create_bow(cls, document_tokens: List[List[str]], dictionary):
        """Create a Bag-of-Words representation of the corpus.

        The BoW vector is a sparse vector representation of the corpus, where each document
        is a list of tuples. Each of these tuples map a document token id to the frequency
        of the token in the document.

        Args:
            document_tokens:
                The corpus as a list of document tokens, where each document is a
                list of candidate keyphrase tokens.
            dictionary:
                A gensim dictionary of the corpus, mapping a unique id to each token.

        Returns:
            A BoW vector (sparse vector) representation of the corpus.
        """
        return [dictionary.doc2bow(document) for document in document_tokens]

    @classmethod
    def train_lda(cls, bow_corpus, dictionary, num_topics: int, eta="auto", alpha="auto",
                  batch_size: Optional[int] = 10, epochs: Optional[int] = 10,
                  iterations: Optional[int] = 400, eval_every: Optional[int] = None):
        """Train the gensim implementation of the LDA model on the corpus.

        Args:
            bow_corpus:
                A BoW vector representation of the corpus.
            dictionary:
                A gensim dictionary of the corpus, mapping a unique id to each token.
            num_topics:
                The amount of latent topics to extract from the corpus (k).
            eta:
                Optional; η - The topic hyperparameter associated with the topic-word distributions. The
                default behavior is to use an asymmetric prior from the corpus.
                β_k ~ Dirichlet(η)
            alpha:
                Optional; α - The proportions hyperparameter associated with the document-topic
                distributions. The default behavior is to learn the asymmetric prior from the data.
                θ_d ~ Dirichlet(α)
            batch_size:
                Optional; The amount of documents to be processed at a time.
            epochs:
                Optional; The amount of complete passes through the dataset before completing training.
            iterations:
                Optional; Maximum iterations on the corpus before inferring a topic distribution.
            eval_every:
                Optional; Evaluate the log perplexity of the model (2x hit to training time).

        Returns:
            An LDA model trained on the corpus.
        """
        return LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=num_topics,
            eta=eta,
            alpha=alpha,
            chunksize=batch_size,
            passes=epochs,
            iterations=iterations,
            eval_every=eval_every)

    @classmethod
    def get_topics(cls, model, num_words: int = 10):
        """Get all of latent topics discovered with the model, as well as the top tokens
        associated with each topic.

        Args:
            model:
                A trained gensim model (LSI/LSA).
            num_words:
                Optional; The amount of words related to each topic to get. The default
                behaviors is to get the top 10 words related to each latent topic.

        Returns:
            A list of topics with associated top keyphrase tokens and keyphrase token significance.
        """
        return model.show_topics(-1, num_words)

    @classmethod
    def print_topics(cls, model, num_words: int = 10) -> None:
        """Print all of the latent topics and top tokens associated with each topic.

        Args:
            model:
                The gensim model for which to calculate the coherence.
            num_words:
                Optional; The amount of words related to each topic to print. The default
                behaviors is to print the top 10 words related to each latent topic.
        """
        topics = TopicModeler.get_topics(model, num_words=num_words)
        topics_dict = []

        for t in topics:
            topics = t[1].replace('"', "").replace('+', "\n").split('\n')
            topics = [topic.strip().split('*') for topic in topics]
            keyphrases = ", ".join([key for _, key in topics])
            topics_dict.append([t[0] + 1, keyphrases])

        df = pd.DataFrame(topics_dict, columns=["Topic #", "Top keywords"])
        df = df.style.set_properties(**{"text-align": "left", "colheader-align": "left"})
        display(df)

    @classmethod
    def get_coherence_umass(cls, model, bow_corpus) -> float:
        """Get the UMass coherence score of the model. Higher is better.

        Args:
            model:
                The gensim model for which to calculate the coherence.
            bow_corpus:
                A BoW vector representation of the corpus.

        Returns:
            The UMass coherence score of the model.
        """
        cm = CoherenceModel(model=model, corpus=bow_corpus, coherence="u_mass")
        return cm.get_coherence()

    @classmethod
    def get_coherence_cv(cls, model, document_tokens, dictionary) -> float:
        """Get the NPMI coherence score of the model. Higher is better.

        Args:
            model:
                The gensim model for which to calculate the coherence.
            document_tokens:
                The corpus as a list of document tokens, where each document is a
                list of candidate keyphrase tokens.
            dictionary:
                A gensim dictionary of the corpus, mapping a unique id to each token.

        Returns:
            The NPMI coherence score of the model.
        """
        cm = CoherenceModel(model=model, texts=document_tokens, dictionary=dictionary,
                            coherence="c_v")
        return cm.get_coherence()

    @classmethod
    def get_perplexity(cls, model, bow_corpus) -> float:
        """Get the log perplexity (per-word likelihood bound) of the model. Higher is better.

        Args:
            model:
                The gensim model for which to calculate the coherence.
            bow_corpus:
                A BoW vector representation of the corpus.

        Returns:
            The log perplexity of the model
        """
        return model.log_perplexity(bow_corpus)


def optimize_num_topics(candidate_num_topics: List[int], corpus: List[List[str]], dictionary, bow_corpus,
                        target: Path, repetitions: int = 3, save: bool = False) -> None:
    """Find the optimal value for the num_topics hyperparameter.

    Given a list of potential values for num_topics, calculate and graph the perplexity
    and coherence scores for each potential values. Calculate and graph the average of
    some number of repetitions.

    Args:
        candidate_num_topics:
            A list of num_topics values to test.
        corpus:
            The corpus as a list of document tokens, where each document is a
            list of candidate keyphrase tokens.
        dictionary:
            A gensim dictionary of the corpus, mapping a unique id to each token.
        bow_corpus:
            A BoW vector representation of the corpus.
        target:
            Path to the location for where to save the graphs.
        repetitions:
            Number of times to train the LDA model on a candidate num_topic before calculating the  average.
        save:
            Save the graphs and trained model to the target location.
    """
    coherence_umass = []
    coherence_cv = []
    perplexity = []

    target = str(target)
    if not os.path.exists(target):
        os.makedirs(target)

    for k in candidate_num_topics:
        logging.info('num_topics=' + str(k))
        cur_coherence_umass = 0
        cur_coherence_cv = 0
        cur_perplexity = 0
        best_coherence_umass = 0
        best_coherence_cv = 0
        best_perplexity = 0

        # Get the average coherences and perplexity over a number of repetitions
        for r in range(repetitions):
            lda_model = TopicModeler.train_lda(bow_corpus, dictionary, num_topics=k)

            cmass = TopicModeler.get_coherence_umass(lda_model, bow_corpus)
            cv = TopicModeler.get_coherence_cv(lda_model, corpus, dictionary)
            p = TopicModeler.get_perplexity(lda_model, bow_corpus)

            cur_coherence_umass += cmass / repetitions
            cur_coherence_cv += cv / repetitions
            cur_perplexity += p / repetitions

            # Save the model that performs the best
            if save and (r == 0 or (cmass > best_coherence_umass
                                    and p > best_perplexity
                                    and cv > best_coherence_cv)):
                lda_model.save(str(Path(target, "num_topics-" + str(k) + ".gensim")))
            lda_model = None

        # Log the avg coherences and perplexity
        coherence_umass.append(cur_coherence_umass)
        coherence_cv.append(cur_coherence_cv)
        perplexity.append(cur_perplexity)
        print("coherence_umass:", coherence_umass)
        print("coherence_cv:", coherence_cv)
        print("perplexity:", perplexity)

    # Visualize the results
    width = 0.35
    xlocs = [i for i in candidate_num_topics]
    target = str(target)

    ## Coherence UMass
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("UMass Coherence for LDA w/ Varying num_topics")
    plt.xlabel("num_topics")
    plt.ylabel("UMass Coherence")

    for x, y in enumerate([round(c, 2) for c in coherence_umass]):
        plt.text(xlocs[x] - width / 2, y + .02, str(y))
    plt.xticks([num + width / 2 for num in candidate_num_topics],
               candidate_num_topics)

    plt.plot(candidate_num_topics, coherence_umass, color='blue', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "num_topics-coherence_umass.png"))

    ## Coherence Cv
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("Cv Coherence for LDA w/ Varying num_topics")
    plt.xlabel("num_topics")
    plt.ylabel("Cv Coherence")

    for x, y in enumerate([round(c, 4) for c in coherence_cv]):
        plt.text(xlocs[x] - width / 2, y, str(y))
    plt.xticks([num + width / 2 for num in candidate_num_topics],
               candidate_num_topics)

    plt.plot(candidate_num_topics, coherence_cv, color='green', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "num_topics-coherence_cv.png"))

    ## Perplexity
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("Perplexity for LDA w/ Varying num_topics")
    plt.xlabel("num_topics")
    plt.ylabel("Perplexity")

    for x, y in enumerate([round(p, 2) for p in perplexity]):
        plt.text(xlocs[x] - width / 2, y + .02, str(y))
    plt.xticks([num + width / 2 for num in candidate_num_topics],
               candidate_num_topics)

    plt.plot(candidate_num_topics, perplexity, color='red', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "num_topics-perplexity.png"))


def optimize_no_above(candidate_no_above: List[int], corpus: List[List[str]], target: Path,
                      no_below: int = 1, repetitions: int = 10, num_topics: int = 8, save: bool = False):
    """Find the optimal value for the no_above hyperparameter.

    Given a list of potential values for no_above, calculate and graph the perplexity
    and coherence scores for each potential values. Calculate and graph the average of
    some number of repetitions.

    Args:
        candidate_no_above:
            A list of no_above values to test.
        corpus:
            The corpus as a list of document tokens, where each document is a
            list of candidate keyphrase tokens.
        target:
            Path to the location for where to save the graphs.
        no_below:
            A fixed integer value for the no_below parameter.
        repetitions:
            Number of times to train the LDA model on a candidate num_topic before calculating the  average.
        num_topics:
            A fixes integer value for the num_topics parameter.
        save:
            Save the graphs and trained model to the target location.
    """
    coherence_umass = []
    coherence_cv = []
    perplexity = []

    target = str(Path)
    if not os.path.exists(target):
        os.makedirs(target)

    for k in candidate_no_above:
        logging.info('no_above=' + str(k))
        cur_coherence_umass = 0
        cur_coherence_cv = 0
        cur_perplexity = 0
        best_coherence_umass = 0
        best_coherence_cv = 0
        best_perplexity = 0

        dictionary = TopicModeler.create_dictionary(corpus, no_below=no_below, no_above=k)
        bow_corpus = TopicModeler.create_bow(corpus, dictionary)

        # Get the average coherences and perplexity over a number of repetitions
        for r in range(repetitions):
            lda_model = TopicModeler.train_lda(bow_corpus, dictionary,
                                               num_topics=num_topics)

            cmass = TopicModeler.get_coherence_umass(lda_model, bow_corpus)
            cv = TopicModeler.get_coherence_cv(lda_model, corpus, dictionary)
            p = TopicModeler.get_perplexity(lda_model, bow_corpus)

            cur_coherence_umass += cmass / repetitions
            cur_coherence_cv += cv / repetitions
            cur_perplexity += p / repetitions

            # Save the model that performs the best
            if save and (r == 0 or (cmass > best_coherence_umass
                                    and p > best_perplexity
                                    and cv > best_coherence_cv)):
                lda_model.save(str(Path(target, "no_above-" + str(k) + ".gensim")))
            lda_model = None

        # Log the avg coherences and perplexity
        coherence_umass.append(cur_coherence_umass)
        coherence_cv.append(cur_coherence_cv)
        perplexity.append(cur_perplexity)
        print("coherence_umass:", coherence_umass)
        print("coherence_cv:", coherence_cv)
        print("perplexity:", perplexity)

    # Visualize the results
    width = 0.01
    xlocs = [i for i in candidate_no_above]
    target = str(target)

    ## UMass Coherence
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("UMass Coherence for LDA w/ Varying no_above")
    plt.xlabel("no_above")
    plt.ylabel("UMass Coherence")

    for x, y in enumerate([round(c, 2) for c in coherence_umass]):
        plt.text(xlocs[x] - width / 2, y + .015, str(y))
    plt.xticks([num + width / 2 for num in candidate_no_above],
               candidate_no_above)

    plt.plot(candidate_no_above, coherence_umass, color='blue', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "no_above-coherence_umass.png"))

    ## Cv Coherence
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("Cv Coherence for LDA w/ Varying no_above")
    plt.xlabel("no_above")
    plt.ylabel("Cv Coherence")

    for x, y in enumerate([round(c, 4) for c in coherence_cv]):
        plt.text(xlocs[x] - width / 2, y, str(y))
    plt.xticks([num + width / 2 for num in candidate_no_above],
               candidate_no_above)

    plt.plot(candidate_no_above, coherence_cv, color='green', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "no_above-coherence_cv.png"))

    ## Perplexity
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("Perplexity for LDA w/ Varying no_above")
    plt.xlabel("no_above")
    plt.ylabel("Perplexity")

    for x, y in enumerate([round(p, 2) for p in perplexity]):
        plt.text(xlocs[x] - width / 2, y + .015, str(y))
    plt.xticks([num + width / 2 for num in candidate_no_above],
               candidate_no_above)

    plt.plot(candidate_no_above, perplexity, color='red', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "no_above-perplexity.png"))


def optimize_no_below(candidate_no_below: List[int], corpus: List[List[str]], target: Path,
                      no_above: float = 0.7, repetitions: int = 10, num_topics: int = 8, save: bool = False):
    """Find the optimal value for the no_below hyperparameter.

    Given a list of potential values for no_below, calculate and graph the perplexity
    and coherence scores for each potential values. Calculate and graph the average of
    some number of repetitions.

    Args:
        candidate_no_below:
            A list of no_below values to test.
        corpus:
            The corpus as a list of document tokens, where each document is a
            list of candidate keyphrase tokens.
        target:
            Path to the location for where to save the graphs.
        no_above:
            A fixed float value for the no_above parameter.
        repetitions:
            Number of times to train the LDA model on a candidate num_topic before calculating the  average.
        num_topics:
            A fixes integer value for the num_topics parameter.
        save:
            Save the graphs and trained model to the target location.
    """
    coherence_umass = []
    coherence_cv = []
    perplexity = []

    target = str(Path)
    if not os.path.exists(target):
        os.makedirs(target)

    for k in candidate_no_below:
        logging.info('no_below=' + str(k))
        cur_coherence_umass = 0
        cur_coherence_cv = 0
        cur_perplexity = 0
        best_coherence_umass = 0
        best_coherence_cv = 0
        best_perplexity = 0

        dictionary = TopicModeler.create_dictionary(corpus, no_below=k, no_above=no_above)
        bow_corpus = TopicModeler.create_bow(corpus, dictionary)

        # Get the average coherences and perplexity over a number of repetitions
        for r in range(repetitions):
            lda_model = TopicModeler.train_lda(bow_corpus, dictionary,
                                               num_topics=num_topics)

            cmass = TopicModeler.get_coherence_umass(lda_model, bow_corpus)
            cv = TopicModeler.get_coherence_cv(lda_model, corpus, dictionary)
            p = TopicModeler.get_perplexity(lda_model, bow_corpus)

            cur_coherence_umass += cmass / repetitions
            cur_coherence_cv += cv / repetitions
            cur_perplexity += p / repetitions

            # Save the model that performs the best
            if save and (r == 0 or (cmass > best_coherence_umass
                                    and p > best_perplexity
                                    and cv > best_coherence_cv)):
                lda_model.save(str(Path(target, "no_below-" + str(k) + ".gensim")))
            lda_model = None

        # Log the avg coherences and perplexity
        coherence_umass.append(cur_coherence_umass)
        coherence_cv.append(cur_coherence_cv)
        perplexity.append(cur_perplexity)
        print("coherence_umass:", coherence_umass)
        print("coherence_cv:", coherence_cv)
        print("perplexity:", perplexity)

    # Visualize the results
    width = 0.01
    xlocs = [i for i in candidate_no_below]
    target = str(target)

    ## UMass Coherence
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("UMass Coherence for LDA w/ Varying no_below")
    plt.xlabel("no_below")
    plt.ylabel("UMass Coherence")

    for x, y in enumerate([round(c, 2) for c in coherence_umass]):
        plt.text(xlocs[x] - width / 2, y + .015, str(y))
    plt.xticks([num + width / 2 for num in candidate_no_below],
               candidate_no_below)

    plt.plot(candidate_no_below, coherence_umass, color='blue', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "no_below-coherence_umass.png"))

    ## Cv Coherence
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("Cv Coherence for LDA w/ Varying no_below")
    plt.xlabel("no_below")
    plt.ylabel("Cv Coherence")

    for x, y in enumerate([round(c, 4) for c in coherence_cv]):
        plt.text(xlocs[x] - width / 2, y, str(y))
    plt.xticks([num + width / 2 for num in candidate_no_below],
               candidate_no_below)

    plt.plot(candidate_no_below, coherence_cv, color='green', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "no_below-coherence_cv.png"))

    ## Perplexity
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.title("Perplexity for LDA w/ Varying no_below")
    plt.xlabel("no_below")
    plt.ylabel("Perplexity")

    for x, y in enumerate([round(p, 2) for p in perplexity]):
        plt.text(xlocs[x] - width / 2, y + .015, str(y))
    plt.xticks([num + width / 2 for num in candidate_no_below],
               candidate_no_below)

    plt.plot(candidate_no_below, perplexity, color='red', linewidth=2, label="LDA")
    plt.legend(loc="best")
    plt.savefig(Path(target, "no_below-perplexity.png"))
