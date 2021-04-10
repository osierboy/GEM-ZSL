import torch
import torch.nn as nn
import torch.nn.functional as F

from GEMZSL.modeling.backbone import resnet101_features
import GEMZSL.modeling.utils as utils

from os.path import join
import pickle


base_architecture_to_features = {
    'resnet101': resnet101_features,
}

class GEMNet(nn.Module):
    def __init__(self, res101, img_size, c, w, h,
                 attritube_num, cls_num, ucls_num, attr_group, w2v,
                 scale=20.0, device=None):

        super(GEMNet, self).__init__()
        self.device = device

        self.img_size = img_size
        # self.prototype_shape = prototype_shape
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.attr_group = attr_group

        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        assert self.w2v_att.shape[0] == self.attritube_num

        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)


        self.backbone = res101

        # self.prototype_vectors = nn.Parameter(nn.init.normal_(torch.empty(self.prototype_shape)), requires_grad=True)  # a, c

        self.W = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
                               requires_grad=True) # 300 * 2048


        self.V = nn.Parameter(nn.init.normal_(torch.empty(self.feat_channel, self.attritube_num)), requires_grad=True)

        # loss
        self.Reg_loss = nn.MSELoss()
        self.CLS_loss = nn.CrossEntropyLoss()



    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        return x

    def base_module(self, x, seen_att):

        N, C, W, H = x.shape
        global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        global_feat = global_feat.view(N, C)
        gs_feat = torch.einsum('bc,cd->bd', global_feat, self.V)

        gs_feat_norm = torch.norm(gs_feat, p=2, dim = 1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)

        temp_norm = torch.norm(seen_att, p=2, dim=1).unsqueeze(1).expand_as(seen_att)
        seen_att_normalized = seen_att.div(temp_norm + 1e-5)

        cos_dist = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        score = cos_dist * self.scale

        return score


    def attentionModule(self, x):

        N, C, W, H = x.shape
        x = x.reshape(N, C, W * H)  # N, V, r=WH

        query = torch.einsum('lw,wv->lv', self.w2v_att, self.W) # L * V

        atten_map = torch.einsum('lv,bvr->blr', query, x) # batch * L * r

        atten_map = F.softmax(atten_map, -1)

        x = x.transpose(2, 1) # batch, WH=r, V
        part_feat = torch.einsum('blr,brv->blv', atten_map, x) # batch * L * V
        part_feat = F.normalize(part_feat, dim=-1)

        atten_map = atten_map.view(N, -1, W, H)
        atten_attr = F.max_pool2d(atten_map, kernel_size=(W,H))
        atten_attr = atten_attr.view(N, -1)

        return part_feat, atten_map, atten_attr, query

    def attr_decorrelation(self, query):

        loss_sum = 0

        for key in self.attr_group:
            group = self.attr_group[key]
            proto_each_group = query[group]  # g1 * v
            channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
            loss_sum += channel_l2_norm.mean()

        loss_sum = loss_sum.float()/len(self.attr_group)

        return loss_sum

    def CPT(self, atten_map):
        """

        :param atten_map: N, L, W, H
        :return:
        """

        N, L, W, H = atten_map.shape
        xp = torch.tensor(list(range(W))).long().unsqueeze(1).to(self.device)
        yp = torch.tensor(list(range(H))).long().unsqueeze(0).to(self.device)

        xp = xp.repeat(1, H)
        yp = yp.repeat(W, 1)

        atten_map_t = atten_map.view(N, L, -1)
        value, idx = atten_map_t.max(dim=-1)

        tx = idx // H
        ty = idx - H * tx

        xp = xp.unsqueeze(0).unsqueeze(0)
        yp = yp.unsqueeze(0).unsqueeze(0)
        tx = tx.unsqueeze(-1).unsqueeze(-1)
        ty = ty.unsqueeze(-1).unsqueeze(-1)

        pos = (xp - tx) ** 2 + (yp - ty) ** 2

        loss = atten_map * pos

        loss = loss.reshape(N, -1).mean(-1)
        loss = loss.mean()

        return loss

    def forward(self, x, att=None, label=None, seen_att=None):

        feat = self.conv_features(x)  # N， 2048， 14， 14

        score = self.base_module(feat, seen_att)  # N, d
        if not self.training:
            return score

        part_feat, atten_map, atten_attr, query = self.attentionModule(feat)

        Lcls = self.CLS_loss(score, label)
        Lreg = self.Reg_loss(atten_attr, att)

        if self.attr_group is not None:
            Lad = self.attr_decorrelation(query)
        else:
            Lad = torch.tensor(0).float().to(self.device)

        Lcpt = self.CPT(atten_map)
        scale = self.scale.item()

        loss_dict = {
            'Reg_loss': Lreg,
            'Cls_loss': Lcls,
            'AD_loss': Lad,
            'CPT_loss': Lcpt,
            'scale': scale
        }

        return loss_dict

    def getAttention(self, x):
        feat = self.conv_features(x)
        part_feat, atten_map, atten_attr, query = self.attentionModule(feat)
        return atten_map


def build_GEMNet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]

    attr_group = utils.get_attr_group(dataset_name)

    img_size = cfg.DATASETS.IMAGE_SIZE


    # res101 feature size
    c,w,h = 2048, img_size//32, img_size//32

    scale = cfg.MODEL.SCALE

    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    model_dir = cfg.PRETRAINED_MODELS

    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)

    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)


    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return GEMNet(res101=res101, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attritube_num=attritube_num,
                  attr_group=attr_group, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)