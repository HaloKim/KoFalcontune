# KoFalcontune

LLM Sota를 갱신한 Falcon을 한국어에 Finetune 할 수 있도록 자료를 공유합니다.

[Faclon](https://huggingface.co/blog/falcon)

|Model|License|Commercial use?|Pretraining length [tokens]|Pretraining compute [PF-days]|Leaderboard score|K,V-cache size for a 2.048 context|
|:----|:----|:----|:----|:----|:----|:----|
|StableLM-Alpha-7B|CC-BY-SA-4.0|✅|1,500B|700|38.3*|800MB|
|LLaMA-7B|LLaMA license|❌|1,000B|500|47.6|1,100MB|
|MPT-7B|Apache 2.0|✅|1,000B|500|48.6|1,100MB|
|Falcon-7B|Apache 2.0|✅|1,500B|700|48.8|20MB|
|LLaMA-33B|LLaMA license|❌|1,500B|3200|56.9|3,300MB|
|LLaMA-65B|LLaMA license|❌|1,500B|6300|58.3|5,400MB|
|Falcon-40B|Apache 2.0|✅|1,000B|2800|60.4|240MB|

# ENV

```
Ubuntu Kuebeflow
A100 80G
```

# Fintune Info

40B 보다 Koalpaca Dataset으로 Fintune 한 경량 7B 모델이 완성에 가까운 데이터를 얻는 것을 확인 할 수 있습니다.

## Data

[KoAlpaca-v1.1](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a/viewer/beomi--KoAlpaca-v1.1a)

## Before

**Model** : Big Falcon Model (40 billion parameters!)

![image](https://github.com/HaloKim/KoFalcontune/assets/44603549/0fb0e7b8-1c43-4417-87a0-63821cc6af66)


## After

**Model** : ybelkada/falcon-7b-sharded-bf16

```python
inputs = tokenizer("광해군은 폭군이었나요 ?", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=512)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

# output

광해군은 폭군이었나요?
광해군은 조선시대에 수많은 외교군들이었습니다. 
그 중에서도 가장 유명한 것은 폭군으로, 그들은 외교정책을 선포하고 신라의 지배를 부담하는 전략을 담당했습니다. 
그러나 광해군은 폭군이지만, 계보형의 외교도 많이 하고 있었습니다. 
그들은 외교문화의 중요성을 인식하고 있었기 때문입니다. 
이들은 외교문화에 맞서 전승전쟁을 전개하였으며, 이란에서 벌였던 외교군의 위업을 대신했습니다. 
그러나 이들은 외교문화에 �    
```

```python
inputs = tokenizer("기계식 키보드 청소방법", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=512)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

# output

기계식 키보드 청소방법은 다음과 같습니다. 
1. 먼저, 키보드에 접착된 물은 무조건 닦아줘야 합니다.
2. 그리고 키보드에 묻은 이쑤시개가 있는 경우, 먼저 진공접근으로 녹인 후 스프레이를 제거합니다.
3. 스프레이를 제거하면 더 깨끗해진다면, 청소기를 사용하는 것이 좋습니다. 청소기를 사용하면 일반적으로 키보드를 청소할 수 있습니다.
4. 마지막으로, 키보드를 뒤집어서 공기가 남지 않게 보충합니다.

그리고 청소기를 사용하는 방법은 다음과 �
```

# wandb 

학습은 10000 step을 진행했습니다.

![image](https://github.com/HaloKim/KoFalcontune/assets/44603549/4041d170-2ae4-4bef-aa28-0dc68101c24a)


<details>
<summary>Origin Git</summary>

# falcontune: 4-Bit Finetuning of FALCONs on a Consumer GPU

**falcontune** allows finetuning FALCONs (e.g., falcon-40b-4bit) on as little as one consumer-grade A100 40GB. 

Its features tiny and easy-to-use codebase.

One benefit of being able to finetune larger LLMs on one GPU is the ability to easily leverage data parallelism for large models.

Underneath the hood, **falcontune** implements the LoRA algorithm over an LLM compressed using the GPTQ algorithm, which requires implementing a backward pass for the quantized LLM.

**falcontune** can generate a 50-token recipe on A100 40GB for ~ 10 seconds using triton backend

```
$ falcontune generate --interactive --model falcon-40b-instruct-4bit --weights gptq_model-4bit--1g.safetensors --max_new_tokens=50 --use_cache --do_sample --prompt "How to prepare pasta?"


How to prepare pasta?
Here's a simple recipe to prepare pasta:

Ingredients:
- 1 pound of dry pasta
- 4-6 cups of water
- Salt (optional)

Instructions:
1. Boil the water

Took 10.042 s
```

This example is based on the model: TheBloke/falcon-40b-instruct-GPTQ.

Here is a [Google Colab](https://colab.research.google.com/drive/1Pv7Dn60u_ANgkhRojAIX-VOkU3J-2cYh?usp=sharing). 
You will need a A100 40GB to finetune the model.

## Installation

### Setup

```
pip install -r requirements.txt 
python setup.py install         
```

The default backend is triton which is the fastest. For cuda support install also the CUDA kernels:

```
python setup_cuda.py install         
```


## Running falcontune

The above process installs a `falcontune` command in your environment.

### Download Models

First, start by downloading the weights of a FALCON model:
```
$ wget https://huggingface.co/TheBloke/falcon-40b-instruct-GPTQ/resolve/main/gptq_model-4bit--1g.safetensors
```

### Generate Text

You can generate text directly from the command line. This generates text from the base model:
```
$ falcontune generate \
    --interactive \
    --model falcon-40b-instruct-4bit \
    --weights gptq_model-4bit--1g.safetensors \
    --max_new_tokens=50 \
    --use_cache \
    --do_sample \
    --instruction "Who was the first person on the moon?"
```

### Finetune A Base Model

You may also finetune a base model yourself. First, you need to download a dataset:
```
$ wget https://github.com/gururise/AlpacaDataCleaned/raw/main/alpaca_data_cleaned.json
```

You can finetune any model of the FALCON family:

<details>
<summary>FALCON-7B</summary>
<br>

    $ falcontune finetune \
        --model=falcon-7b \
        --weights=tiiuae/falcon-7b \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-7b-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-7b-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-7b \
        --weights tiiuae/falcon-7b \
        --lora_apply_dir falcon-7b-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>


<details>
<summary>FALCON-7B-INSTRUCT</summary>
<br>

    $ falcontune finetune \
        --model=falcon-7b-instruct \
        --weights=tiiuae/falcon-7b-instruct \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-7b-instruct-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-7b-instruct-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-7b-instruct \
        --weights mosaicml/falcon-7b-instruct \
        --lora_apply_dir falcon-7b-instruct-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>


<details>
<summary>FALCON-40B</summary>
<br>

    $ falcontune finetune \
        --model=falcon-40b \
        --weights=tiiuae/falcon-40b \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-40b-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-40b-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-40b \
        --weights tiiuae/falcon-40b\
        --lora_apply_dir falcon-40b-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>

<details>
<summary>FALCON-40B-INSTRUCT</summary>
<br>

    $ falcontune finetune \
        --model=falcon-40b-instruct \
        --weights=tiiuae/falcon-40b-instruct \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-40b-instruct-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-40b-instruct-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-40b-instruct \
        --weights tiiuae/falcon-40b-instruct\
        --lora_apply_dir falcon-40b-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>

<details>
<summary>FALCON-7B-INSTRUCT-4BIT</summary>
<br>

    $ wget https://huggingface.co/TheBloke/falcon-7b-instruct-GPTQ/resolve/main/gptq_model-4bit-64g.safetensors
    
    $ falcontune finetune \
        --model=falcon-7b-instruct-4bit \
        --weights=gptq_model-4bit-64g.safetensors \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-7b-instruct-4bit-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-7b-instruct-4bit-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-7b-instruct-4bit \
        --weights gptq_model-4bit-64g.safetensors \
        --lora_apply_dir falcon-7b-instruct-4bit-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>

<details>
<summary>FALCON-40B-INSTRUCT-4BIT</summary>
<br>

    $ wget https://huggingface.co/TheBloke/falcon-40b-instruct-GPTQ/resolve/main/gptq_model-4bit--1g.safetensors
    
    $ falcontune finetune \
        --model=falcon-40b-instruct-4bit \
        --weights=gptq_model-4bit--1g.safetensors \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-40b-instruct-4bit-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-40b-instruct-4bit-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-40b-instruct-4bit \
        --weights gptq_model-4bit--1g.safetensors \
        --lora_apply_dir falcon-40b-instruct-4bit-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>

## Acknowledgements

**falcontune** is based on the following projects:
* The GPTQ algorithm and codebase by the [IST-DASLAB](https://github.com/IST-DASLab/gptq) with modifications by [@qwopqwop200](https://github.com/qwopqwop200/)
* The `alpaca_lora_4bit` repo by [johnsmith0031](https://github.com/johnsmith0031)
* The PEFT repo and its implementation of LoRA
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium
* The `llmtune` repo by [kuleshov-group](https://github.com/kuleshov-group/llmtune)


## Consultations
Need a custom solution? Let me know: `r.m.mihaylov@gmail.com`
