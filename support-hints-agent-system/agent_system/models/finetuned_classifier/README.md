---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:23973
- loss:CosineSimilarityLoss
widget:
- source_sentence: Подскажите, пожалуйста, Как скачать и настроить мобильное приложение?
    Подробно, пожалуйста.
  sentences:
  - 'Подскажите алгоритм: Какая доходность? Если можно — по пунктам.'
  - 'Нужна помощь: За сколько одобряется экспресс-кредит?'
  - 'Прошу подсказать: Первый вход в Интернет-банк?'
- source_sentence: 'Нужна инструкция: Какая доходность? Если можно — кратко.'
  sentences:
  - 'Подскажите по шагам: Карта заблокирована - что делать? Заранее спасибо.'
  - 'Нужна помощь: Как скачать и настроить мобильное приложение? Кратко, пожалуйста.'
  - Что делать если банк обанкротился? Что такое возмещение по вкладам? Подробно,
    пожалуйста.
- source_sentence: 'Подскажите алгоритм: Что входит в пакет услуг Infinite? Пожалуйста.'
  sentences:
  - Будьте добры, Как получить карту Форсаж?
  - 'Подскажите коротко: На какие товары распространяется? Если можно — кратко.'
  - 'Интересует вопрос: На какой срок вклад СуперСемь? Если можно — кратко.'
- source_sentence: 'Подскажите алгоритм: Рекомендации по безопасному использованию
    карточек? Буду признателен.'
  sentences:
  - Не подскажете, Какой регламент размещения онлайн-депозитов? Спасибо.
  - 'Подскажите алгоритм: Я потерял пин-код к карточке. Как его восстановить? Если
    можно — кратко.'
  - 'Нужна помощь: Какие устройства совместимы с Mir Pay? Заранее спасибо.'
- source_sentence: Подскажите, пожалуйста, Как активировать карту MORE? Кратко, пожалуйста.
  sentences:
  - 'Нужна инструкция: А логин и пароль в новом Интернет-банке и Мобильном банке отличаются?
    Кратко, пожалуйста.'
  - 'Подскажите порядок: Кто может оформить карту Отличник? Заранее спасибо.'
  - 'Подскажите по шагам: Могу ли я закрыть вклад, размещенный в офисе банка ВТБ (Беларусь)
    расположенном в Минске, находясь в другом городе? Заранее спасибо.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Подскажите, пожалуйста, Как активировать карту MORE? Кратко, пожалуйста.',
    'Подскажите порядок: Кто может оформить карту Отличник? Заранее спасибо.',
    'Нужна инструкция: А логин и пароль в новом Интернет-банке и Мобильном банке отличаются? Кратко, пожалуйста.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9998, -0.0059],
#         [ 0.9998,  1.0000, -0.0038],
#         [-0.0059, -0.0038,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 23,973 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 9 tokens</li><li>mean: 21.89 tokens</li><li>max: 63 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 21.97 tokens</li><li>max: 63 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.67</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                             | sentence_1                                                                                               | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Подскажите порядок: Каким способом можно пополнить карточку, оформленную в Банке ВТБ (Беларусь), находясь в Москве? Подробно, пожалуйста.</code> | <code>Подскажите, Почему не получается войти под логином и паролем из старого ДБО? Спасибо.</code>       | <code>1.0</code> |
  | <code>Будьте добры, Рекомендации по безопасному использованию карточек? Заранее спасибо.</code>                                                        | <code>Хочу уточнить: Что делать если банк обанкротился? Что такое возмещение по вкладам? Спасибо.</code> | <code>1.0</code> |
  | <code>Подскажите алгоритм: Система пишет, что такого клиента нет. Что делать? Пожалуйста.</code>                                                       | <code>Подскажите, Могу ли я оформить онлайн-вклад в вашем банке? Буду признателен.</code>                | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.1668 | 500   | 0.0787        |
| 0.3337 | 1000  | 0.0031        |
| 0.5005 | 1500  | 0.0002        |
| 0.6673 | 2000  | 0.0002        |
| 0.8342 | 2500  | 0.0001        |
| 1.0010 | 3000  | 0.0001        |
| 1.1678 | 3500  | 0.0001        |
| 1.3347 | 4000  | 0.0001        |
| 1.5015 | 4500  | 0.0001        |
| 1.6683 | 5000  | 0.0           |
| 1.8352 | 5500  | 0.0           |
| 2.0020 | 6000  | 0.0001        |
| 2.1688 | 6500  | 0.0           |
| 2.3357 | 7000  | 0.0008        |
| 2.5025 | 7500  | 0.0001        |
| 2.6693 | 8000  | 0.0           |
| 2.8362 | 8500  | 0.0           |
| 3.0030 | 9000  | 0.0           |
| 3.1698 | 9500  | 0.0           |
| 3.3367 | 10000 | 0.0           |
| 3.5035 | 10500 | 0.0           |
| 3.6703 | 11000 | 0.0           |
| 3.8372 | 11500 | 0.0           |


### Framework Versions
- Python: 3.12.0
- Sentence Transformers: 5.1.1
- Transformers: 4.57.1
- PyTorch: 2.6.0+cu124
- Accelerate: 1.10.1
- Datasets: 4.2.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->