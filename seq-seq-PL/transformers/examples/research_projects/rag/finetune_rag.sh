# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag.sh --help to see all the possible options


#--cache_dir /home/gsir059/Videos/Custom-Hug/transformers/examples/research_projects/rag/cache_dir \
python /home/gsir059/Videos/Custom-Hug-New/transformers/examples/research_projects/rag/finetune_rag.py  \
    --data_dir /home/gsir059/Videos/Custom-Hug-New/transformers/examples/research_projects/rag/data \
    --output_dir /home/gsir059/Videos/Custom-Hug/transformers/examples/research_projects/rag/outputs \
    --model_name_or_path facebook/bart-large  \
    --model_type bart \
    --gpus 1 \
    --do_train \
    --do_predict \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --val_max_target_length 128 \
    --test_max_target_length 128 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 1 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1\
    --index_name custom\
    --passages_path /home/gsir059/Videos/Custom-Hug-New/transformers/examples/research_projects/rag/my_knowledge_dataset/my_knowledge_dataset\
    --index_path  /home/gsir059/Videos/Custom-Hug-New/transformers/examples/research_projects/rag/my_knowledge_dataset/my_knowledge_dataset_hnsw_index.faiss
   

