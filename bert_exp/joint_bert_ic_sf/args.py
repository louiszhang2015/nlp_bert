
arg = {

    "seed": 1234,
    "no_cuda": True,
    "model_type": "bert",
    "task": "snips",
    "data_dir": "data/jointBert",
    "intent_label_file": "intent_label.txt",
    "slot_label_file": "slot_label.txt",
    "max_seq_len": 50,
    "ignore_index": 0,
    "dropout_rate": 0.1,
    "use_crf": True,
    "slot_loss_coef": 1.0,
    "max_steps": -1,
    "num_train_epochs": 10.0,
    "weight_decay": 0.0,
    "learning_rate": 5e-5,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0,
    "num_train_epochs": 10,
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "logging_steps": 200,
    "save_steps": 200,
    "max_grad_norm": 1.0,
    "eval_batch_size": 64,



}
