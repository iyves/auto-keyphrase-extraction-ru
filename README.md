## Undergraduate honors thesis
<hr />
<div align="center">

### AUTOMATIC KEYPHRASE EXTRACTION FROM RUSSIAN-LANGUAGE SCHOLARLY PAPERS IN COMPUTATIONAL LINGUISTICS
_Yves Wienecke_<br/>
Thesis advisor: _William Comer, Ph.D_
</div><br/>

__Abstract.__ The automatic extraction of keyphrases from scholarly papers is a necessary step for many Natural Language Processing 
(NLP) tasks, including text retrieval, machine translation, and text summarization. However, due to the different 
grammatical and semantic intricacies of languages, this is a highly language-dependent task. Many free and open source 
implementations of state-of-the-art keyphrase extraction techniques exist, but they are not adapted for processing 
Russian text. Furthermore, the multi-linguistic character of scholarly papers in the field of Russian computational 
linguistics and NLP introduces additional complexity to keyphrase extraction. This is a free and open 
source program as a proof of concept for a topic-clustering approach to the automatic extraction of keyphrases from the 
largest conference on Russian computational linguistics and intellectual technologies, Dialogue. The goal 
is to use LDA and pyLDAvis to discover the latent topics of the Dialogue conference and to extract the salient 
keyphrases used by the research community. The conclusion points to needed improvements to techniques for PDF text 
extraction, morphological normalization, and candidate keyphrase ranking.

__Keywords:__ Automatic Keyphrase Extraction, Topic Modeling, LDA, pyLDAvis, Scholarly Papers, Russian.
<br/><br/>

### Running the code 
In the `src/` directory, there are three [Jupyter Notebooks](https://jupyter.org/), `corpus_creation.ipynb`, 
`preprocessing.ipynb`, and `topic_modeling.ipynb`. For ease, it is recommend running the programs from the notebooks. 
Download or clone this repository to your local machine. Then, cd into the `src` directory and type
```sh
jupyter notebook
```
from a terminal. This should open Jupyter and allow you to open one of the notebooks. 
Once you have opened a notebook, navigate to a `cell` and press the `Run` button to run the code.
