import torch
import numpy as np
from sklearn.metrics import accuracy_score

def cal_accuracy(model, dataloadr, att, test_id, device, bias=None):

    scores = []
    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, label) in enumerate(dataloadr):
        img = img.to(device)
        score = model(img, seen_att=att)
        scores.append(score)
        labels.append(label)

    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    if bias is not None:
        scores = scores-bias

    _,pred = scores.max(dim=1)
    pred = pred.view(-1).to(cpu)

    outpred = test_id[pred]
    outpred = np.array(outpred, dtype='int')
    labels = labels.numpy()
    unique_labels = np.unique(labels)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(labels == l)[0]
        acc += accuracy_score(labels[idx], outpred[idx])
    acc = acc / unique_labels.shape[0]
    return acc

def eval(
        tu_loader,
        ts_loader,
        att_unseen,
        att_seen,
        cls_unseen_num,
        cls_seen_num,
        test_id,
        train_test_id,
        model,
        test_gamma,
        device
):

    acc_zsl = cal_accuracy(model=model, dataloadr=tu_loader, att=att_unseen, test_id=test_id, device=device, bias=None)

    bias_s = torch.zeros((1, cls_seen_num)).fill_(test_gamma).to(device)
    bias_u = torch.zeros((1, cls_unseen_num)).to(device)
    bias = torch.cat([bias_s, bias_u], dim=1)

    att = torch.cat((att_seen, att_unseen), dim=0)
    acc_gzsl_unseen = cal_accuracy(model=model, dataloadr=tu_loader, att=att,
                                   test_id=train_test_id, device=device, bias=bias)
    acc_gzsl_seen = cal_accuracy(model=model, dataloadr=ts_loader, att=att,
                                   test_id=train_test_id, device=device,bias=bias)
    H = 2 * acc_gzsl_seen * acc_gzsl_unseen / (acc_gzsl_seen + acc_gzsl_unseen)

    return acc_zsl, acc_gzsl_unseen, acc_gzsl_seen, H


def eval_zs_gzsl(
        tu_loader,
        ts_loader,
        res,
        model,
        test_gamma,
        device
):
    model.eval()
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)

    test_id = res['test_id']
    train_test_id = res['train_test_id']

    cls_seen_num = att_seen.shape[0]
    cls_unseen_num = att_unseen.shape[0]

    with torch.no_grad():
        acc_zsl, acc_gzsl_unseen, acc_gzsl_seen, H = eval(
            tu_loader,
            ts_loader,
            att_unseen,
            att_seen,
            cls_unseen_num,
            cls_seen_num,
            test_id,
            train_test_id,
            model,
            test_gamma,
            device
        )

    model.train()

    return acc_gzsl_seen, acc_gzsl_unseen, H, acc_zsl