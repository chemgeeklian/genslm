Install genslm in a new conda env:

```sh
conda create -n genslm_agent python=3.12
conda activate genslm_agent

git clone https://github.com/chemgeeklian/genslm
cd genslm
pip install -e .

pip install 'accelerate>=0.26.0'
```

Change `train_path`, `eval_path`, `model_name`, `model_cache_dir` accordingly in `config.yaml`

Then

```sh
cd examples/training/finetune_fasta
python finetune_genslm.py --config config.yaml 
```