# Continuation
We use the (BBQ)[https://arxiv.org/abs/2110.08193] and (CBBQ)[https://arxiv.org/abs/2306.16244] dataset for bias evaluation. We subsample part of the dataset for evaluation. We synthesize the spoken context using three public available TTS systems for each languages: Azure and Google for both Chinese and English QA, Amazon polly for English QA, Topmediai for Chinese QA. 

## Dataset
The datasets are available from [huggingface dataset](https://huggingface.co/SLLMBias). 

## Evaluation
Inference your SILLM with the dataset, and save the results in CSV file with this format
```
id,tts_provider,speaker,prediction_sent
1_ambiguous,azure,zh-CN-XiaoxiaoMultilingualNeural,没有足够信息
1_ambiguous,azure,zh-CN-YunxiNeural,没有足够信息
1_ambiguous,google,cmn-CN-Wavenet-A,没有足够信息
1_ambiguous,google,cmn-CN-Wavenet-C,没有足够信息
1_ambiguous,tts_maker,hao_郝维,没有足够信息
1_ambiguous,tts_maker,MX-文君,没有足够信息
...
```
And then run the evaluation script:
```
python3 eval.py -f <path_to_csv>
```
Our experiment result in CSV format are available from `./predictions`