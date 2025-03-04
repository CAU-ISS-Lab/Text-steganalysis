import os
import pickle

from classifier.classifier_getter import get_classifier
from train.train import train
from train.test_t import test

from tools.tool import parse_args, print_args, set_seed

import dataset.loader as loader
from embedding.embedding import get_embedding
import warnings
warnings.filterwarnings("ignore")
import datetime
from termcolor import colored


def main():

    # make_print_to_file(path='/results')

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    args.id2word = vocab.itos
    #print(test_data)

    # initialize model
    model = {}
    model["G"], model["D"] = get_embedding(vocab, args)
    model["clf"] = get_classifier(model["G"].ebd_dim, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train(train_data, val_data, model, args)

    # val_acc, val_std, _ = test(val_data, model, args,
    #                                         args.val_episodes)

        args.mode = "test"
    #print(args.mode,"MODE: ")

    test_acc, test_std, test_pre,test_re, drawn_data = test(test_data, model, args,
                                          args.test_episodes)

    total = sum([param.nelement() for param in model["G"].parameters()])+sum([param.nelement() for param in model["D"].parameters()])+sum([param.nelement() for param in model["clf"].parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))


    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
            "test_pre": test_pre,
            "test_re": test_re,
        }
        domain = args.Comments
        shot=args.shot
        now_time = datetime.datetime.now()
        now_time =datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')

        '''for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)'''
        with open(args.result_path, "a") as f:
            f.write("  "+now_time + "  shot: " + str(shot) + "  query: " + str(args.query)+"  way: " + str(args.way)+ '\n')
            f.write("  domain: " + domain + "  ebd: " + args.embedding+"  classifier: " + args.classifier+"   domain:  "+ domain+'\n')
            f.write("  test_acc: " + str(test_acc) + "   test_acc: " + str(test_acc) + '\n')
            f.write("  test_pre: " + str(test_pre) + "   test_re: " + str(test_re) + '\n')
            f.write('\r\n')
            f.write('\r\n')
            f.write('\r\n')
            f.write('\r\n')

        print(colored("Test results are stored in the result.txt file!", "yellow"))



if __name__ == '__main__':
    main()

