# Coreference resolution
We use the dev set of [WinoBias](https://uclanlp.github.io/corefBias/overview) dataset, and the spoken utterance are from [WinoST](https://aclanthology.org/2022.lrec-1.230/).

## Dataset
The datasets are available from [zenodo](https://zenodo.org/records/4139080). 

## Evaluation
Inference your SILLM with the dataset, and save the results in JSON file with this format
```
{
    "396": "janitor",
    "398": "Not known",
    "399": "assistant",
    "400": "assistant",
    ...
}
```
And then run the evaluation script:
```
python3 eval.py -f <path_to_json>
```