## Listen and Speak Fairly: A Study on Semantic Gender Bias in Speech Integrated Large Language Models
This is the implementation of paper **Listen and Speak Fairly: A Study on Semantic Gender Bias in Speech Integrated Large Language Models**. We evaluate gender bias in SILLMs across four semantic-related tasks: speech-to-text translation (STT), spoken coreference resolution (SCR), spoken sentence continuation (SSC), and spoken question answering (SQA). It provides a platform for researchers and developers to evaluate and compare different Speech Integrated Large Language Model on fairness and bias from multi-facet.

![Listen and Speak Fairly overview](docs/speech_llm_bias.png)

### Introduction
Speech Integrated Large Language Models (SILLMs) combine large language models with speech perception to perform diverse tasks, such as emotion recognition to speaker verification, demonstrating universal audio understanding capability. However, these models may amplify biases present in training data, potentially leading to biased access to information for marginalized groups. This work introduces a curated spoken bias evaluation toolkit and corresponding dataset.

### Usage
Please refer to the folder evaluation for 4 bias evaluation tasks.

### Contact
If you have any concerns or suggestions, please contact: even.dlion8@gmail.com

### Citation
If our work assists you, please consider cite our paper.
```
@misc{lin2024listenspeakfairlystudy,
      title={Listen and Speak Fairly: A Study on Semantic Gender Bias in Speech Integrated Large Language Models}, 
      author={Yi-Cheng Lin and Tzu-Quan Lin and Chih-Kai Yang and Ke-Han Lu and Wei-Chih Chen and Chun-Yi Kuan and Hung-yi Lee},
      year={2024},
      eprint={2407.06957},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2407.06957}, 
}
```