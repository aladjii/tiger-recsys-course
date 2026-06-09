# Generative Retrieval on Yambda (TIGER + RQ-VAE)

RQ-VAE / RQ-KMeans / RQ-OPQ tokenizers and a TIGER-style encoder-decoder for sequential recommendation on [Yambda-500M](https://huggingface.co/datasets/yandex/yambda). SASRec and popularity baselines use the vendored Yambda benchmark code.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_yambda.py --prepare-listens-fltrd
```

`prepare_listens_fltrd` filters flat listens with `timestamp > 18_000_000` and `played_ratio_pct > 90`, then writes:

- `dataset/yambda/flat/500m/listens_fltrd.parquet`
- `dataset/yambda/sequential/500m/listens_fltrd.parquet`

Training caches processed tensors in `dataset/yambda/processed/` (created automatically).

## Train

Gin configs in `configs/`. Pass one config path as the only argument.

**Tokenizers**

```bash
python train_rqvae.py configs/rqvae_listens_fltrd.gin
python train_rq_tokenizer.py configs/rq_kmeans_listens_fltrd.gin
python train_rq_tokenizer.py configs/rq_opq_listens_fltrd.gin
```

**Decoder** (set `train.pretrained_rqvae_path` in the gin file to the tokenizer checkpoint)

```bash
python train_decoder.py configs/decoder_listens_fltrd_25k_dedup.gin   # RQ-VAE
python train_decoder.py configs/decoder_listens_fltrd_25k_kmeans.gin
python train_decoder.py configs/decoder_listens_fltrd_25k_opq.gin
```

**Collision experiments** (OPQ tokenizer variants)

```bash
python train_rq_tokenizer.py configs/rq_opq_hamr_listens_fltrd.gin
python train_decoder.py configs/decoder_listens_fltrd_25k_opq_hamr.gin
python train_decoder.py configs/decoder_listens_fltrd_25k_opq_pop.gin
python train_decoder.py configs/decoder_listens_fltrd_25k_opq_rrs.gin
```

## Eval

```bash
python eval_decoder.py \
  --checkpoint out/decoder/listens_fltrd_opq/checkpoint_24999.pt \
  --pretrained_rqvae_path out/rq_opq/listens_fltrd/checkpoint_19999.pt \
  --filter_seen
```

For popularity / RRS decoders add `--corpus_assignment popularity` or `--corpus_assignment rrs --rrs_k 4`.

**SASRec**

```bash
python benchmarks/models/sasrec/train.py \
  --exp_name sasrec_listens_fltrd \
  --data_dir dataset/yambda \
  --size 500m \
  --interaction listens_fltrd

python benchmarks/models/sasrec/eval.py \
  --exp_name sasrec_listens_fltrd \
  --data_dir dataset/yambda \
  --size 500m \
  --interaction listens_fltrd \
  --filter_seen
```

## Layout

```
configs/          gin configs
data/             Yambda dataset loaders
modules/          tokenizers, TIGER model, collision utilities
benchmarks/       Yambda eval code + SASRec / popularity baselines
scripts/          download + data preparation
eval_decoder.py   TIGER eval with filter_seen
docs/             experiment report
```

`out/`, `logs/`, `dataset/` are gitignored. Keep large artifacts outside the repo if needed.

## Report

`docs/yambda500m_generative_retrieval_report.md`

## References

- Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023
- [Yambda-5B](https://huggingface.co/datasets/yandex/yambda)
- Forked from [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
