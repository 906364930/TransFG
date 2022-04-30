import collections

import torch


def combine_for_TransFG(weights_ori):
    weights = {}
    weights['model'] = collections.OrderedDict()
    weights['model'] = weights_ori['model'].copy()
    PART_ATTENTION_Q_WEIGHT = 'transformer.encoder.part_layer.attn.query.weight'
    PART_ATTENTION_Q_BIAS = 'transformer.encoder.part_layer.attn.query.bias'
    PART_ATTENTION_K_WEIGHT = 'transformer.encoder.part_layer.attn.key.weight'
    PART_ATTENTION_K_BIAS = 'transformer.encoder.part_layer.attn.key.bias'
    PART_ATTENTION_V_WEIGHT = 'transformer.encoder.part_layer.attn.value.weight'
    PART_ATTENTION_V_BIAS = 'transformer.encoder.part_layer.attn.value.bias'
    # PART_ATTENTION_Tvq_WEIGHT = 'transformer.encoder.part_layer.attn.Tvq.weight'
    # PART_ATTENTION_Tvk_WEIGHT = 'transformer.encoder.part_layer.attn.Tvk.weight'
    PART_ATTENTION_Tvq_Tvkt = 'transformer.encoder.part_layer.attn.Tvq_vkT'

    Wq = weights['model'][PART_ATTENTION_Q_WEIGHT]
    biasQ = weights['model'][PART_ATTENTION_Q_BIAS].unsqueeze(0)
    Wk = weights['model'][PART_ATTENTION_K_WEIGHT]
    biasK = weights['model'][PART_ATTENTION_K_BIAS].unsqueeze(0)
    Wv = weights['model'][PART_ATTENTION_V_WEIGHT]
    biasV = weights['model'][PART_ATTENTION_V_BIAS].unsqueeze(0)
    W_newQ = torch.cat((Wq, biasQ), dim=0)  # [769,768]
    W_newK = torch.cat((Wk, biasK), dim=0)  # [769,768]
    W_newV = torch.cat((Wv, biasV), dim=0)  # [769,768]

    Tvq = compute_Transfer_Matrix(W_newV, W_newQ)
    Tvk = compute_Transfer_Matrix(W_newV, W_newK)
    Tvq_Tvkt = torch.matmul(Tvq, Tvk.transpose(-1, -2))
    weights['model'][PART_ATTENTION_Tvq_Tvkt] = Tvq_Tvkt

    # weights['model'][PART_ATTENTION_Tvq_WEIGHT] = Tvq
    # weights['model'][PART_ATTENTION_Tvk_WEIGHT] = Tvk
    # check(W_newQ, W_newV, Tvq)

    del weights['model'][PART_ATTENTION_Q_WEIGHT]
    del weights['model'][PART_ATTENTION_Q_BIAS]
    del weights['model'][PART_ATTENTION_K_WEIGHT]
    del weights['model'][PART_ATTENTION_K_BIAS]

    for i_head in range(11):
        ATTENTION_Q_WEIGHT = 'transformer.encoder.layer.{}.attn.query.weight'.format(i_head)
        ATTENTION_Q_BIAS = 'transformer.encoder.layer.{}.attn.query.bias'.format(i_head)
        ATTENTION_K_WEIGHT = 'transformer.encoder.layer.{}.attn.key.weight'.format(i_head)
        ATTENTION_K_BIAS = 'transformer.encoder.layer.{}.attn.key.bias'.format(i_head)
        ATTENTION_V_WEIGHT = 'transformer.encoder.layer.{}.attn.value.weight'.format(i_head)
        ATTENTION_V_BIAS = 'transformer.encoder.layer.{}.attn.value.bias'.format(i_head)
        # ATTENTION_Tvq = 'transformer.encoder.layer.{}.attn.Tvq.weight'.format(i_head)
        # ATTENTION_Tvk = 'transformer.encoder.layer.{}.attn.Tvk.weight'.format(i_head)
        ATTENTION_Tvq_Tvkt = 'transformer.encoder.layer.{}.attn.Tvq_vkT'.format(i_head)
        Wq = weights['model'][ATTENTION_Q_WEIGHT]  # [768,768]
        biasQ = weights['model'][ATTENTION_Q_BIAS].unsqueeze(0)  # [1,768]
        Wk = weights['model'][ATTENTION_K_WEIGHT]  # [768,768]
        biasK = weights['model'][ATTENTION_K_BIAS].unsqueeze(0)  # [1,768]
        Wv = weights['model'][ATTENTION_V_WEIGHT]  # [768,768]
        biasV = weights['model'][ATTENTION_V_BIAS].unsqueeze(0)  # [1,768]
        W_newQ = torch.cat((Wq, biasQ), dim=0)  # [769,768]
        W_newK = torch.cat((Wk, biasK), dim=0)  # [769,768]
        W_newV = torch.cat((Wv, biasV), dim=0)  # [769,768]
        Tvq = compute_Transfer_Matrix(W_newV, W_newQ)
        Tvk = compute_Transfer_Matrix(W_newV, W_newK)
        Tvq_Tvkt = torch.matmul(Tvq, Tvk.transpose(-1, -2))
        weights['model'][ATTENTION_Tvq_Tvkt] = Tvq_Tvkt
        # weights['model'][ATTENTION_Tvq] = Tvq
        # weights['model'][ATTENTION_Tvk] = Tvk

        del weights['model'][ATTENTION_Q_WEIGHT]
        del weights['model'][ATTENTION_Q_BIAS]
        del weights['model'][ATTENTION_K_WEIGHT]
        del weights['model'][ATTENTION_K_BIAS]

    return weights

    # Wq = weights['transformer.encoder.layer.0.attn.query.weight']    # [768,768]
    # biasQ = weights['transformer.encoder.layer.0.attn.query.bias'].unsqueeze(0)   # [1,768]
    # W_newQ = torch.cat((Wq, biasQ), dim=0)  # [769,768]
    # Wk = weights['transformer.encoder.layer.0.attn.key.weight']    # [768,768]
    # biasK = weights['transformer.encoder.layer.0.attn.key.bias'].unsqueeze(0)   # [1,768]
    # W_newK = torch.cat((Wk, biasK), dim=0)  # [769,768]
    # Wc = torch.matmul(W_newQ, W_newK.transpose(-1, -2))  # [769,769]
    # sample = torch.randn((197, 768), device='cuda')
    #
    # Q = torch.matmul(sample, Wq)+biasQ.squeeze(0)
    # K = torch.matmul(sample, Wk)+biasK.squeeze(0)
    #
    # res1 = torch.matmul(Q, K.transpose(-1, -2))
    # p = torch.ones(sample.shape[0], device='cuda')  # [197]
    # p = p.unsqueeze(1)   # [197,1]
    # sample = torch.cat((sample, p), dim=1)    # [197,769]
    # res2 = torch.matmul(torch.matmul(sample, Wc), sample.transpose(-1, -2))
    # print(res1==res2)
    # return weights_new


def compute_Transfer_Matrix(Wv, Wx):
    """
    give weight Wv Wx,and compute the transfer matrix,the number of transfer matrix is
    equals to the number of transformer's head(12)
    Tvx1:inv(transpose(Wv1)*Wv1)*transpose(Wv1)*Wx1
    ....
    Tvxi:inv(transpose(Wvi)*Wvi)*transpose(Wvi)*Wxi
    :param Wv: [769,768]
    :param Wx: [769,768]
    :return Tvx: [12,64,64]
    """
    new_v_shape = Wv.size()[:-1] + (12, 64)  # [769,12,64]
    Wv = Wv.view(*new_v_shape)  # [769,12,64]
    Wv = Wv.permute(1, 0, 2)  # [12,769,64]
    new_x_shape = Wx.size()[:-1] + (12, 64)  # [769,12,64]
    Wx = Wx.view(*new_x_shape)  # [12,769,64]
    Wx = Wx.permute(1, 0, 2)  # [12,769,64]
    Wv_transpose = Wv.transpose(-1, -2)  # [12,64,769]
    temp_matrix = torch.matmul(Wv_transpose, Wv)  # [12,64,64]
    # a = temp_matrix.inverse() @ temp_matrix
    # a = a.cpu().numpy()

    Tvx = temp_matrix.inverse() @ Wv_transpose @ Wx  # [12,64,64]

    return Tvx


def check(W_newx, W_newv, T):
    """
    judge if W_newx is equals to W_newv@T
    :param W_new: [769,768]
    :param W_newv: [769,768]
    :param T: # [12,64,64]
    :return correct/wrong str
    """
    new_v_shape = W_newv.size()[:-1] + (12, 64)  # [769,12,64]
    Wv = W_newv.view(*new_v_shape)  # [769,12,64]
    Wv = Wv.permute(1, 0, 2)  # [12,769,64]
    Wv_transpose = Wv.transpose(-1, -2)  # [12,64,769]
    # W_temp = torch.matmul(Wv, T)  # [12,769,64]
    Wx = W_newx.view(*new_v_shape)  # [769,12,64]
    Wx = Wx.permute(1, 0, 2)  # [12,769,64]
    right = Wv_transpose @ Wx  # [12,64,64]
    left = Wv_transpose @ Wv @ T  # [12,64,64]

    temp_see1 = (abs(right - left)).cpu().numpy()


if __name__ == "__main__":
    weights_ori = torch.load("/home/luoqinglu/TransFg-with-git/weight/sample_run_checkpoint.bin")
    weights_new = combine_for_TransFG(weights_ori)
    print('weight transposed')
    torch.save(weights_new, "/home/luoqinglu/TransFg-with-git/weight/sample_run_checkpoint_combine_qkv.bin")
