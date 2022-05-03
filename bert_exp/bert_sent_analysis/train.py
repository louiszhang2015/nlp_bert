from bert_exp.bert_sent_analysis.data_loader import load_train_data, load_test_data
from bert_exp.bert_sent_analysis.data_process import find_max_length


from pprint import pprint
from sklearn.model_selection import train_test_split
import torch


def set_up_gpu():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


def train_model():

    # Get the data
    train_data = load_train_data()
    sents = train_data.tweet.values
    label = train_data.label.values

    sents_train, sents_val, label_train, label_val = train_test_split(sents, label, test_size=0.1, random_state=2020)
    test_data = load_test_data()

    # set up the GPU
    set_up_gpu()

    find_max_length(train_data, test_data)
    import pdb; pdb.set_trace()
