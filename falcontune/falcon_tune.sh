pip install -r requirements.txt
python setup.py install

# urllib3 에러 발생시
pip install 'urllib3<2'

falcontune finetune \
    --model=falcon-7b-instruct \
    --weights=tiiuae/falcon-7b-instruct \
    --dataset=./KoAlpaca_v1.1.json \
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
