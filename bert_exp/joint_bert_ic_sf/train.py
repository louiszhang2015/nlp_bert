from bert_exp.joint_bert_ic_sf.args import arg
from bert_exp.joint_bert_ic_sf.utils import init_logger, set_seed, load_tokenizer
from bert_exp.joint_bert_ic_sf.data_loader.load_data import load_and_cache_examples
from bert_exp.joint_bert_ic_sf.trainer import Trainer



def train_model():
    set_seed()
    init_logger()
    tokenizer = load_tokenizer()

    train_dataset = load_and_cache_examples(tokenizer, mode="train")
    dev_dataset   = load_and_cache_examples(tokenizer, mode="dev")
    test_dataset  = load_and_cache_examples(tokenizer, mode="test")

    trainer = Trainer(train_dataset, dev_dataset, test_dataset)

    trainer.train()
