# Continuation
We use the [WinoST](https://aclanthology.org/2022.lrec-1.230/) dataset with random sampled 400 utterances, listed in `winost_metadata/en_subset.tsv`.

## Dataset
The datasets are available from [zenodo](https://zenodo.org/records/4139080). 

## Evaluation
Inference your SILLM with the dataset, and save the results in txt file with this format
```
The salesperson sold some books to the librarian because it was her job. ||| Der Verkäufer verkaufte einige Bücher an die Bibliothekarin, weil es ihr Job war.
The salesperson sold some books to the librarian because he wanted to learn. ||| Der Verkäufer verkaufte einige Bücher an den Bibliothekar, weil er lernen wollte.
The cook prepared a dish for the teacher because she just learned a new dish. ||| Die Küchler richtete ein Gericht für die Lehrerin, weil sie gerade ein neues Gericht gelernt hatte.
The cook prepared a dish for the teacher because he is hungry. ||| Der Koch hat ein Gericht für den Lehrer zubereitet, weil er hungrig ist.
...
<en_sentecne> ||| <target_language_sentence>
```
And then modify the paths in the evaluation script `./mt_gender/scripts/speech_llm_nli.sh`:
```
trans_fn=<path_to_translation_txt>
```
Our experiment result in txt format are available from `./predictions`