import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from transformers import AdamW
from sklearn import metrics


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer \
            if not any(nd in n for nd in no_decay)], 'weight':0.01},
        {'params': [p for n, p in param_optimizer \
            if any(nd in n for nd in no_decay)], 'weight':0.0}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        warmup=0.05,
        t_total=len(train_iter)*args.epochs)

    steps = 0
    best_loss = 99 
    last_step = 0
    model.train()

    for epoch in range(1, args.epochs+1):
        print('\n--------training epochs: {}-----------'.format(epoch))
        print(args.train_stego_dir)
        for batch in train_iter:
            feature, target = batch[0], batch[1]
            
            optimizer.zero_grad()
            torch.cuda.synchronize()
            logit = model(feature)
            torch.cuda.synchronize()
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data \
                    == target.data).sum()
                accuracy = corrects.item()/args.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss:{:.6f} acc:{:.4f}({}/{})'.format(
                    steps, loss.item(), accuracy, corrects, args.batch_size))
		
            if steps % args.test_interval == 0:
                dev_acc, dev_loss = data_eval(dev_iter, model, args)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best.pt')
                if epoch>10 and dev_loss > 0.9:
                    print('\nthe validation is {}, training done...'\
                        .format(dev_loss))
                    sys.exit(0)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                model.train()

			# elif steps % args.save_interval == 0:
			# 	save(model, args.save_dir, 'snapshot', steps)


def data_eval(data_loader, model, args):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    logits_list = []
    targets_list = []

    with torch.no_grad():
        for batch in data_loader:
            (input_ids, seq_len, attention_mask), labels = batch
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)


            # 使用你的MyBert模型进行预测
            outputs = model(input_ids, attention_mask=attention_mask)

            logits = outputs

            if labels is not None:
                correct_predictions += (logits.argmax(dim=-1) == labels).sum().item()
                targets_list.extend(labels.cpu().tolist())

            logits_list.append(logits.cpu())

    # 聚合所有batch的结果
    all_logits = torch.cat(logits_list, dim=0)
    predictions = all_logits.argmax(dim=-1).numpy()  # 不需要额外的.cpu()调用，因为已经在添加到list时调用
    targets_array = np.array(targets_list)

    # 计算性能指标
    accuracy = metrics.accuracy_score(targets_array, predictions) if len(targets_list) > 0 else 0
    precision = metrics.precision_score(targets_array, predictions, average='binary') if len(targets_list) > 0 else 0
    recall = metrics.recall_score(targets_array, predictions, average='binary') if len(targets_list) > 0 else 0
    F1_score = metrics.f1_score(targets_array, predictions, average='weighted') if len(targets_list) > 0 else 0

    TN, FP, FN, TP = metrics.confusion_matrix(targets_array, predictions).ravel()

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {F1_score:.4f}')
    with open('test_results.txt', 'a', errors='ignore') as f:
        f.write(f'{args.save_dir}\n')
        f.write(f'The testing accuracy: {accuracy:.4f}\n')
        f.write(f'The testing precision: {precision:.4f}\n')
        f.write(f'The testing recall: {recall:.4f}\n')
        f.write(f'The testing F1_score: {F1_score:.4f}\n')
        f.write(f'The testing TN: {TN}\n')
        f.write(f'The testing TP: {TP}\n')
        f.write(f'The testing FN: {FN}\n')
        f.write(f'The testing FP: {FP}\n\n')


def save(model, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_prefix)
    torch.save(model.state_dict(), save_path)
			
