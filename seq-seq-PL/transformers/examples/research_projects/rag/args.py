from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"



class GenericArgs():
    def __init__(self,root_dir):
        self.root_dir=root_dir


    @staticmethod
    def add_data_specific_args(parser):
    
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )
        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument(
            "--max_source_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--prefix",
            type=str,
            default=None,
            help="Prefix added at the beginning of each text, typically used with T5-based models.",
        )
        parser.add_argument(
            "--n_train", 
            type=int, 
            default=-1, 
            required=False, 
            help="# examples. -1 means use all."
        )
        parser.add_argument(
            "--n_val", 
            type=int, 
            default=10, 
            required=False, 
            help="# examples. -1 means use all."
        )
        parser.add_argument(
            "--n_test", 
            type=int, 
            default=10, 
            required=False, 
            help="# examples. -1 means use all."
        )

        parser.add_argument(
            "--sortish_sampler",
            action="store_true",
            help="Whether to Sortish Samler or not. It sorts the inputs according to\
            lenghts in-order to minimizing the padding size.")

        return parser



    @staticmethod
    def add_generic_model_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--model_type",
            choices=["rag_sequence", "rag_token", "bart", "t5"],
            type=str,
            help="RAG model type: sequence or token, if none specified, the type is inferred from the model_name_or_path",
        )
        parser.add_argument(
            "--config_name", 
            default="", type=str, 
            help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from huggingface.co",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )


        return parser

    def add_training_args(parser):

        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument(
            "--fp16_opt_level",
            type=str,
            default="O2",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html",
        )
        parser.add_argument(
            "--n_tpu_cores", 
            dest="tpu_cores", 
            type=int
        )
        parser.add_argument(
            "--max_grad_norm", 
            dest="gradient_clip_val", 
            default=1.0, 
            type=float, 
            help="Max gradient norm"
        )
        parser.add_argument(
            "--do_train",
                action="store_true",
                help="Whether to run training."
        )
        parser.add_argument(
            "--do_predict", 
            action="store_true", 
            help="Whether to run predictions on the test set."
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            dest="accumulate_grad_batches",
            type=int,
            default=2,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--seed", 
            type=int, 
            default=42, 
            help="random seed for initialization"
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            "--distributed-port", 
            type=int, default=-1, 
            required=False, 
            help="Port number for distributed training."
        )  
        parser.add_argument(
            "--logger_name", 
            type=str, 
            choices=["default", "wandb", "tb_logs","wandb_shared"], 
            default="default"
        )
        parser.add_argument(
            "--label_smoothing", 
            type=float, 
            default=0.0, 
            required=False
        )


        parser.add_argument(
            "--learning_rate", 
            default=5e-5, 
            type=float, 
            help="The initial learning rate for Adam."
        )
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument(
            "--weight_decay", 
            default=0.0, 
            type=float, 
            help="Weight decay if we apply some."
        )
        parser.add_argument(
            "--adam_epsilon", 
            default=1e-8, 
            type=float, 
            help="Epsilon for Adam optimizer."
        )
        parser.add_argument(
            "--warmup_steps", 
            default=0, 
            type=int, 
            help="Linear warmup over warmup_steps."
        )
        parser.add_argument(
            "--num_workers", 
            default=4, 
            type=int, 
            help="kwarg passed to DataLoader"
        )
        parser.add_argument(
            "--num_train_epochs", 
            dest="max_epochs", 
            default=3, 
            type=int
        )
        parser.add_argument(
            "--train_batch_size", 
            default=32, 
            type=int
        )
        parser.add_argument(
            "--eval_batch_size", 
            default=32, 
            type=int
        )
        parser.add_argument(
            "--adafactor", 
            action="store_true"
        )

        parser.add_argument(
            "--profile",
            action="store_true",
            help="If True, use pytorch_lightning.profiler.AdvancedProfiler to profile the Trainer.",
        )

        parser.add_argument(
            "--local_rank",  
            default=-1, 
            type=int, 
            help="For distributed training: local_rank"
        )

        return parser



