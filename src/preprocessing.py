from gensim.models import Phrases
from helpers import get_file_names, get_text
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
import matplotlib.pyplot as plt
import nltk
import os
import re
import rus_preprocessing_udpipe

nltk.download("stopwords")
stopwordsiso = get_text("stopwords-ru.txt").split("\n")
from nltk.corpus import stopwords
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ScholarlyPreprocessor(object):
    """Prepares a list of raw Russian text from scholarly parpers into a list of normalized 
    tokens.
    """
    
    russian_stopwords = stopwords.words("russian") + stopwords.words("english") + \
        stopwordsiso + ["что-то", "который", "это", "также", "диалог", "что-ловек", "чем-ловек", "как-то",
                       "поскольку", "никак", "текст", "явление", "являться", "автор", "вообще-то", "получать",
                       "сравнивать", "корпус", "исследование", "словарь", "конструкция", "таблица", "предложение",
                       "эксперимент", "причина", "отношение", "данные", "объект", "анализ", "рисяча", "во-вторых",
                       "во-первых", "в-третьих", "заключение", "выражение", "высказывание", "материал", 
                       "использовать", "чей", "например", "тема", "форма", "прилагательное", "глагол", "плюс",
                       "какой-то", "чем-л", "что-л", "субъект", "употребление", "приставка", "смысл", "ситуация",
                       "частица", "контекст", "речь", "речевой", "термин", "шаблон", "существительное", "жест",
                       "создавать", "перевод", "группа", "участник", "система", "наречие", "лексема",
                       "модель", "союз", "существовать", "местоимение"]
    is_sentence_delimiter = re.compile(r'^[\.!?\n]$')
    
    # Regular expressions for filtering out from text
    references_section = re.compile(r'Литература.*$|Список литературы.*$', flags=re.IGNORECASE | re.MULTILINE)
    parentheses_or_brackets = re.compile(r'\(.*?\)|\[.*?\]', flags=re.DOTALL)
    word_at_pagebreak = re.compile(r'([А-яЁё\w]+-[\n$])|\n([а-яёa-z]+)')
    
    # Regular expressions for filtering tokens to keep
    length_threshold = re.compile(r'^.{3,100}$')
    has_alphabetical = re.compile(r'^[А-яЁё0-9]+(?:-[А-яЁё0-9]+)*$')
    not_number_token = re.compile(r'^(?![x0-9]+$).*$')
    has_alphabetical_or_is_sentence_delimiter = re.compile(r'{}|{}'.format(
        has_alphabetical.pattern, is_sentence_delimiter.pattern))

    
    @classmethod
    def remove_references_section(cls, text: str, file_name: Optional[Path] = None) -> str:
        """Remove any test past the last line that ends in either 'Литература' 
        or 'Список литературы'.
        
        Args:
            text:
                The full text of a scholarly paper.
            file_name: 
                Optional; The path to the file that contains the text that is being
                preprocessed.

        Returns:
            The text of a scholarly paper except for the references section. 
        """
        # Get the last match
        match = None
        for match in re.finditer(cls.references_section, text):
            pass

        if match == None:
            logging.info("Could not find references section" +
                         ("" if (file_name == None) else (" for file: " + str(file_name))))
        else:
            text = text[:match.span()[0]]
        return text
    
    
    @classmethod
    def remove_short_paragraphs(cls, text: str, min_paragraph_length: int = 100):
        """Remove any paragraphs that are shorter than a specified length.
        
        Args:
            text:
                The full text of a scholarly paper.
            min_paragraph_length:
                Optional; The threshold for deciding how long a paragraph
                must be to not be removed. The default is to remove paragraphs shorter than
                100 chars.
        
        Returns:
            The text of a scholarly paper except for any paragraphs shorter than a specified
            length
        """
        paragraphs = [" ".join(paragraph.split("\n")) for paragraph in text.split("\n\n")]
        paragraphs = [paragraph for paragraph in paragraphs 
                      if len(paragraph) >= min_paragraph_length]
        return '\n'.join(paragraphs)
    
    
    @classmethod
    def filter_text(cls, text: str, patterns: List[re.Pattern]) -> str:
        """Filter substrings from a text based on a list of regular expressions.
        
        Args:
            text: 
                The text to filter.
            patterns: 
                A list of compiled regex patterns to match.
            
        Returns:
            Filtered text that doesn't match the list of regular expressions.
        """
        filtered_text = text
        for pattern in patterns:
            filtered_text = re.sub(pattern, "", filtered_text)
        return filtered_text
    
    
    @classmethod
    def tokenize(cls, text: str, keep_pos: bool = False, 
                 keep_punct: bool = True) -> List[List[str]]:
        """Lemmatize and tokenize a text paragraph by paragraph with udpipe.
        
        Args:
            text:
                The text to tokenize. Paragraphs are delimited by a single newline character
            keep_pos:
                Optional; Append the part of speech tag to the end of each token or
                not. The default behavior is to drop the POS tag.
            keep_punct:
                Optional; Keep punctuation marks as separate tokens. The default
                behavior is to keep punctuation.
        
        Returns:
            A list of each paragraph, where each paragraph is a list of lemmatized tokens. 
        """
        tokens: List[List[str]] = [rus_preprocessing_udpipe.process(
            rus_preprocessing_udpipe.process_pipeline, text=paragraph,
            keep_pos=keep_pos, keep_punct=keep_punct) + ['\n']
                  for paragraph in text.split('\n')]
 
        return tokens
    
    
    @classmethod
    def lowercase_tokens(cls, tokens: List[str]) -> List[str]:
        """Convert any tokens to its lowercase equivalent.
        
        Args:
            text: 
                The list of tokens to lowercase.
        
        Returns:
            A list of lowercase tokens.
        """
        return [token.lower() for token in tokens]
    
    
    @classmethod
    def filter_tokens(cls, tokens: List[str], patterns: List[re.Pattern]) -> List[str]:
        """Filter out tokens that don't match a list of regular expressions.
        
        Args:
            tokens:
                The list of tokens to filter.
            patterns: 
                A list of compiled regex patterns to match.
        
        Returns:
            A list of tokens which match the list of regular expressions.
        """
        filtered_tokens = tokens
        for pattern in patterns:
            filtered_tokens = [token for token in filtered_tokens
                               if re.search(pattern, token)]
        return filtered_tokens
    
    
    @classmethod
    def remove_stop_words(cls, tokens: List[str], stop: List[str]) -> List[str]:
        """Remove frequently appearing words from a text.
        
        Args:
            tokens:
                The list of tokens from which to remove stop words.
            stop:
                The list of frequently appearing words.
        
        Returns:
            The list of tokens minus all lone stop words.
        """
        return [token for token in tokens 
                if not token in stop]
    

    @classmethod
    def get_ngrams(cls, tokens: List[List[str]], n: int = 2, min_count: int = 3,
                   delimiter: str = b' ', stop: Optional[List[str]] = None) -> List[List[str]]:
        """Add up to tri-grams to a list of tokens.
        
        Args:
            tokens:
                The list of paragraph tokens from which to search for ngrams.
            n:
                Optional, either '2' or '3'; Up to bigrams or trigrams. The default is to
                add up to bigrams.
            min_count: 
                Optional; The minimum amount of occurances for an ngram to be 
                added. The default is to add ngrams that occur at least 3 times.
            delimiter:
                Optional; The byte string to separate words in an n-gram. The
                default is to separate words in an n-gram with a space.
            stop:
                Optional; A list of stop words.
        
        Returns:
            A list of sentence tokens plus ngrams.
        """
        # Break down the list of paragraph tokens into a list of sentences tokens
        tokens = [token for paragraph in tokens for token in paragraph]
        sentences = [list(token) for delimiter, token in
                     groupby(tokens, lambda token: re.match(cls.is_sentence_delimiter, token))
                     if not delimiter]
        amt_sentences = len(sentences)
        
        # Find the bigrams
        bigram = Phrases(sentences, min_count=min_count, delimiter=delimiter,
                        common_terms=stop)

        if n == 3:
            # Find the trigrams
            trigram = Phrases(bigram[sentences], min_count=1, delimiter=delimiter,
                             common_terms=stop)
            for sentence in range(amt_sentences):
                sentences[sentence] = [n_gram for n_gram in trigram[bigram[sentences[sentence]]]]
        else:
            for sentence in range(amt_sentences):
                sentences[sentence] = [n_gram for n_gram in bigram[sentences[sentence]]]
                
        return sentences
    
    
    @classmethod
    def preprocess_one(cls, document: str, file_name: Optional[Path] = None, 
                       verbose: bool = False) -> List[str]:
        """Preprocess a single raw scholarly text into keyphrase candidates.
        
        Args:
            document:
                The document to preprocess
            file_name:
                Optional; The name of the file that contains the document.
            verbose:
                Optional; If true, prints out the document after each preprocessing step. The
                default behavior is to not do this.
        
        Returns:
            A list of candidate keyphrases from the document.
        """
        if (verbose):
            print("-"*50, "\nRaw text, before any preprocessing:\n\n")
            [[print(line) for line in paragraph.split("\n")]
             for paragraph in document.split("\n\n")]
            print("\n\n", "-"*50)
            
        document = cls.remove_references_section(document, file_name)
        document = cls.remove_short_paragraphs(document)

        # Remove any words at page breaks, bracketed text, and parethesized text
        document = cls.filter_text(document, [cls.parentheses_or_brackets, cls.word_at_pagebreak])
        if verbose:
            print("\nAfter removing the references section, short paragraphs, "
                  "parenthesized/bracketed text, and text at pagebreaks:\n\n")
            [print(line) for line in document.split('\n')]
            print("\n\n", "-"*50)
        
        # After tokenization, the document is represented as a list of paragraph tokens,
        # where each paragraph is a list of tokens
        document: List[List[str]] = cls.tokenize(document)
        if verbose:
            print("\nAfter tokenization:\n\n")
            [print(line) for line in document]
            print("\n\n", "-"*50)
        
        document = [cls.filter_tokens(paragraph, [cls.has_alphabetical_or_is_sentence_delimiter])
                    for paragraph in document]
        if verbose:
            print("\nAfter removing tokens which do not have at least 1 alphabetical character "
                  "and are not sentence delimiters:\n\n")
            [print(line) for line in document]
            print("\n\n", "-"*50)
        
        # After getting n-grams, the document is represented as a list of sentence tokens,
        # where each sentence is a list of ngrams.
        document = cls.get_ngrams(document, n=3, stop=cls.russian_stopwords)
        if verbose:
            print("\nAfter adding bigrams and trigrams:\n\n")
            [print(sentence) for sentence in document]
            print("\n\n", "-"*50)
        
        document = [cls.filter_tokens(sentence, [cls.length_threshold, 
                                                 cls.not_number_token])
                    for sentence in document]
        document = [cls.remove_stop_words(sentence, cls.russian_stopwords)
                    for sentence in document]
        if verbose:
            print("\nAfter filtering short tokens, long tokens, and stop words:\n\n")
            [print(sentence) for sentence in document]
            print("\n\n", "-"*50)
        
        # Flatten the sentence tokens, so that the document is represented as a list of 
        # candidate keyphrases
        document = [token for sentence in document for token in sentence]
        if verbose:
            print("\nFinal preprocessed text:\n\n", document, "\n\n", "-"*50)
        return document
        
    
    @classmethod
    def preprocess(cls, text: List[str], file_names: Optional[List[Path]] = None) -> List[List[str]]:
        """Preprocess a corpus of raw scholarly text into keyphrase candidates for each of
        the documents in the corpus.
        
        Args:
            text:
                The corpus as a list of documents as raw text.
            file_names:
                Optional; A list corrsponding the file names for each document in the corpus.
        
        Returns:
            The corpus as lists of keyphrase candidate tokens for each document.
        """
        if (file_names == None):
            text = [cls.preprocess_one(document) for document in text]
        else:
            text = [cls.preprocess_one(document, file_name) 
                    for document, file_name in zip(text, file_names)]
        return text