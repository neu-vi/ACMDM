# A modified version of "Fully Convolutional Mesh Autoencoder using Efficient Spatially Varying Kernels"
# https://arxiv.org/abs/2006.04325
# and thanks to this more modern implementation as well
# https://github.com/g-fiche/Mesh-VQ-VAE
# https://arxiv.org/abs/2312.08291
import torch
import torch.nn as nn
import numpy as np
import os

#################################################################################
#                                       AE                                      #
#################################################################################
class AE(nn.Module):
    def __init__(self, model, bs=16, num_vertices=6890):
        super().__init__()
        # currently only set up is for SMPL-H
        self.num_vertices = num_vertices
        self.bs=bs
        self.encoder = Encoder(model)
        self.decoder = Decoder(model)

    def encode(self, x):
        B, L = x.shape[0], x.shape[1]
        x = x.view(B * L, self.num_vertices, 3)
        x_encoder = self.encoder(x)
        return x_encoder

    def forward(self, x):
        B, L = x.shape[0], x.shape[1]
        x = x.view(B * L, self.num_vertices, 3)
        x_encoder = self.encoder(x)
        x_out = self.decoder(x_encoder)
        x_out = x_out.view(B, L, self.num_vertices, 3)
        return x_out


    def decode(self, x):
        T = x.shape[1]
        if x.shape[1] % self.bs != 0:
            x = torch.cat([x, torch.zeros_like(x[:, :self.bs-x.shape[1] % self.bs])], dim=1)
        outputs = []
        for i in range(x.shape[0]):
            outputss = []
            for j in range(0, x.shape[1], self.bs):
                chunk = x[i, j:j + self.bs]
                out = self.decoder(chunk)
                outputss.append(out)
            outputs.append(torch.cat(outputss, dim=0)[:T])
        x_out = torch.stack(outputs, dim=0)

        return x_out

#################################################################################
#                                     AE Zoos                                   #
#################################################################################
def ae(**kwargs):
    config_model = {"batch": 16,
                    "connection_folder": "body_models/ConnectionMatrices/",
                    "initial_connection_fn": "body_models/ConnectionMatrices/_pool0.npy",
                    "connection_layer_lst": ["pool0", "pool1", "pool2", "pool3", "pool4", "pool5", "pool6", "pool7_28",
                                             "unpool7_28", "unpool6", "unpool5", "unpool4", "unpool3", "unpool2",
                                             "unpool1", "unpool0"],
                    "channel_lst": [64, 64, 128, 128, 256, 256, 512, 12, 512, 256, 256, 128, 128, 64, 64, 3],
                    "weight_num_lst": [9, 0, 9, 0, 9, 0, 9, 0, 0, 9, 0, 9, 0, 9, 0, 9],
                    "residual_rate_lst": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                    }
    return AE(FullyConvAE(config_model, **kwargs), bs=config_model["batch"])
AE_models = {
    'AE_Model': ae
}


class Encoder(nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model.forward_till_layer_n(x, len(self.model.channel_lst) // 2)
        return out


class Decoder(nn.Module):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model.forward_from_layer_n(x, len(self.model.channel_lst) // 2)
        return out

class FullyConvAE(nn.Module):
    def __init__(
        self, config_model=None, test_mode=False
    ):  # layer_info_lst= [(point_num, feature_dim)]
        super(FullyConvAE, self).__init__()

        self.test_mode = test_mode

        self.channel_lst = config_model["channel_lst"]

        self.residual_rate_lst = config_model["residual_rate_lst"]

        self.weight_num_lst = config_model["weight_num_lst"]

        self.initial_connection_fn = config_model["initial_connection_fn"]

        data = np.load(self.initial_connection_fn)
        neighbor_id_dist_lstlst = data[:, 1:]  # point_num*(1+2*neighbor_num)
        self.point_num = data.shape[0]
        self.neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape(
            (self.point_num, -1, 2)
        )[
            :, :, 0
        ]  # point_num*neighbor_num
        self.neighbor_num_lst = np.array(data[:, 0])  # point_num

        self.relu = nn.ELU()

        self.batch = config_model["batch"]

        #####For Laplace computation######
        self.initial_neighbor_id_lstlst = torch.LongTensor(
            self.neighbor_id_lstlst
        ).cuda()  # point_num*max_neighbor_num
        self.initial_neighbor_num_lst = torch.FloatTensor(
            self.neighbor_num_lst
        ).cuda()  # point_num

        self.connection_folder = config_model["connection_folder"]
        self.connection_layer_fn_lst = []
        fn_lst = os.listdir(self.connection_folder)
        self.connection_layer_lst = config_model["connection_layer_lst"]
        for layer_name in self.connection_layer_lst:
            layer_name = "_" + layer_name + "."

            find_fn = False
            for fn in fn_lst:
                if (layer_name in fn) and ((".npy" in fn) or (".npz" in fn)):
                    self.connection_layer_fn_lst += [self.connection_folder + fn]
                    find_fn = True
                    break
            if find_fn == False:
                print("!!!ERROR: cannot find the connection layer fn")

        self.init_layers(self.batch)

        self.initial_max_neighbor_num = self.initial_neighbor_id_lstlst.shape[1]

    def init_layers(self, batch):
        self.layer_lst = (
            []
        )  ##[in_channel, out_channel, in_pn, out_pn, max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst,conv_layer, residual_layer]

        self.layer_num = len(self.channel_lst)

        in_point_num = self.point_num
        in_channel = 3

        for l in range(self.layer_num):
            out_channel = self.channel_lst[l]
            weight_num = self.weight_num_lst[l]
            residual_rate = self.residual_rate_lst[l]

            connection_info = np.load(self.connection_layer_fn_lst[l])
            out_point_num = connection_info.shape[0]
            neighbor_num_lst = torch.FloatTensor(
                connection_info[:, 0].astype(float)
            ).cuda()  # out_point_num*1
            neighbor_id_dist_lstlst = connection_info[
                :, 1:
            ]  # out_point_num*(max_neighbor_num*2)
            print(self.connection_layer_fn_lst[l])
            print()
            neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape(
                (out_point_num, -1, 2)
            )[
                :, :, 0
            ]  # out_point_num*max_neighbor_num
            neighbor_id_lstlst = torch.LongTensor(neighbor_id_lstlst).cuda()
            max_neighbor_num = neighbor_id_lstlst.shape[1]
            avg_neighbor_num = round(neighbor_num_lst.mean().item())
            effective_w_weights_rate = neighbor_num_lst.sum() / float(
                max_neighbor_num * out_point_num
            )
            effective_w_weights_rate = round(effective_w_weights_rate.item(), 3)

            pc_mask = torch.ones(in_point_num + 1).cuda()
            pc_mask[in_point_num] = 0
            neighbor_mask_lst = pc_mask[
                neighbor_id_lstlst
            ].contiguous()  # out_pn*max_neighbor_num neighbor is 1 otherwise 0

            zeros_batch_outpn_outchannel = torch.zeros(
                (batch, out_point_num, out_channel)
            ).cuda()

            if (residual_rate < 0) or (residual_rate > 1):
                print("Invalid residual rate", residual_rate)
            ####parameters for conv###############
            conv_layer = ""

            if residual_rate < 1:
                weights = torch.randn(weight_num, out_channel * in_channel).cuda()

                weights = nn.Parameter(weights).cuda()

                self.register_parameter("weights" + str(l), weights)

                bias = nn.Parameter(torch.zeros(out_channel).cuda())
                self.register_parameter("bias" + str(l), bias)

                w_weights = torch.randn(out_point_num, max_neighbor_num, weight_num) / (
                    avg_neighbor_num * weight_num
                )

                w_weights = nn.Parameter(w_weights.cuda())
                self.register_parameter("w_weights" + str(l), w_weights)

                conv_layer = (weights, bias, w_weights)

            ####parameters for residual###############

            ## a residual layer with out_point_num==in_point_num and residual_rate==1 is a pooling or unpooling layer

            residual_layer = ""

            if residual_rate > 0:
                p_neighbors = ""
                weight_res = ""

                if out_point_num != in_point_num:
                    p_neighbors = nn.Parameter(
                        (
                            torch.randn(out_point_num, max_neighbor_num)
                            / (avg_neighbor_num)
                        ).cuda()
                    )
                    self.register_parameter("p_neighbors" + str(l), p_neighbors)

                if out_channel != in_channel:
                    weight_res = torch.randn(out_channel, in_channel)
                    # self.normalize_weights(weight_res)
                    weight_res = weight_res / out_channel
                    weight_res = nn.Parameter(weight_res.cuda())
                    self.register_parameter("weight_res" + str(l), weight_res)

                residual_layer = (weight_res, p_neighbors)

            #####put everythin together

            layer = (
                in_channel,
                out_channel,
                in_point_num,
                out_point_num,
                weight_num,
                max_neighbor_num,
                neighbor_num_lst,
                neighbor_id_lstlst,
                conv_layer,
                residual_layer,
                residual_rate,
                neighbor_mask_lst,
                zeros_batch_outpn_outchannel,
            )

            self.layer_lst += [layer]

            in_point_num = out_point_num
            in_channel = out_channel

    # precompute the parameters so as to accelerate forwarding in testing mode
    def init_test_mode(self):
        for l in range(len(self.layer_lst)):
            layer_info = self.layer_lst[l]

            (
                in_channel,
                out_channel,
                in_pn,
                out_pn,
                weight_num,
                max_neighbor_num,
                neighbor_num_lst,
                neighbor_id_lstlst,
                conv_layer,
                residual_layer,
                residual_rate,
                neighbor_mask_lst,
                zeros_batch_outpn_outchannel,
            ) = layer_info

            if len(conv_layer) != 0:
                (
                    weights,
                    bias,
                    raw_w_weights,
                ) = conv_layer  # weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num

                w_weights = ""

                w_weights = raw_w_weights * neighbor_mask_lst.view(
                    out_pn, max_neighbor_num, 1
                ).repeat(
                    1, 1, weight_num
                )  # out_pn*max_neighbor_num*weight_num

                weights = torch.einsum(
                    "pmw,wc->pmc", [w_weights, weights]
                )  # out_pn*max_neighbor_num*(out_channel*in_channel)
                weights = weights.view(
                    out_pn, max_neighbor_num, out_channel, in_channel
                )

                conv_layer = weights, bias

            ####compute output of residual layer####

            if len(residual_layer) != 0:
                (
                    weight_res,
                    p_neighbors_raw,
                ) = residual_layer  # out_channel*in_channel  out_pn*max_neighbor_num
                if in_pn != out_pn:
                    p_neighbors = torch.abs(p_neighbors_raw) * neighbor_mask_lst
                    p_neighbors_sum = p_neighbors.sum(1) + 1e-8  # out_pn
                    p_neighbors = p_neighbors / p_neighbors_sum.view(out_pn, 1).repeat(
                        1, max_neighbor_num
                    )

                    residual_layer = weight_res, p_neighbors

            self.layer_lst[l] = (
                in_channel,
                out_channel,
                in_pn,
                out_pn,
                weight_num,
                max_neighbor_num,
                neighbor_num_lst,
                neighbor_id_lstlst,
                conv_layer,
                residual_layer,
                residual_rate,
                neighbor_mask_lst,
                zeros_batch_outpn_outchannel,
            )

    # a faster mode for testing
    # input_pc batch*in_pn*in_channel
    # out_pc batch*out_pn*out_channel
    def forward_one_conv_layer_batch_during_test(
        self, in_pc, layer_info, is_final_layer=False
    ):
        batch = in_pc.shape[0]

        (
            in_channel,
            out_channel,
            in_pn,
            out_pn,
            weight_num,
            max_neighbor_num,
            neighbor_num_lst,
            neighbor_id_lstlst,
            conv_layer,
            residual_layer,
            residual_rate,
            neighbor_mask_lst,
            zeros_batch_outpn_outchannel,
        ) = layer_info

        device = in_pc.get_device()
        if device < 0:
            device = "cpu"

        in_pc_pad = torch.cat(
            (in_pc, torch.zeros(batch, 1, in_channel).to(device)), 1
        )  # batch*(in_pn+1)*in_channel

        in_neighbors = in_pc_pad[
            :, neighbor_id_lstlst.to(device)
        ]  # batch*out_pn*max_neighbor_num*in_channel

        ####compute output of convolution layer####
        out_pc_conv = zeros_batch_outpn_outchannel.clone()

        if len(conv_layer) != 0:
            (
                weights,
                bias,
            ) = conv_layer  # weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num

            out_neighbors = torch.einsum(
                "pmoi,bpmi->bpmo", [weights.to(device), in_neighbors]
            )  # batch*out_pn*max_neighbor_num*out_channel

            out_pc_conv = out_neighbors.sum(2)

            out_pc_conv = out_pc_conv + bias

            if is_final_layer == False:
                out_pc_conv = self.relu(
                    out_pc_conv
                )  ##self.relu is defined in the init function

        # if(self.residual_rate==0):
        #    return out_pc
        ####compute output of residual layer####
        out_pc_res = zeros_batch_outpn_outchannel.clone()

        if len(residual_layer) != 0:
            (
                weight_res,
                p_neighbors,
            ) = residual_layer  # out_channel*in_channel  out_pn*max_neighbor_num

            if in_channel != out_channel:
                in_pc_pad = torch.einsum("oi,bpi->bpo", [weight_res, in_pc_pad])

            out_pc_res = []
            if in_pn == out_pn:
                out_pc_res = in_pc_pad[:, 0:in_pn].clone()
            else:
                in_neighbors = in_pc_pad[
                    :, neighbor_id_lstlst.to(device)
                ]  # batch*out_pn*max_neighbor_num*out_channel
                out_pc_res = torch.einsum(
                    "pm,bpmo->bpo", [p_neighbors.to(device), in_neighbors]
                )

        out_pc = out_pc_conv.to(device) * np.sqrt(1 - residual_rate) + out_pc_res.to(
            device
        ) * np.sqrt(residual_rate)

        return out_pc

    # use in train mode. Slower than test mode
    # input_pc batch*in_pn*in_channel
    # out_pc batch*out_pn*out_channel
    def forward_one_conv_layer_batch(self, in_pc, layer_info, is_final_layer=False):
        batch = in_pc.shape[0]

        (
            in_channel,
            out_channel,
            in_pn,
            out_pn,
            weight_num,
            max_neighbor_num,
            neighbor_num_lst,
            neighbor_id_lstlst,
            conv_layer,
            residual_layer,
            residual_rate,
            neighbor_mask_lst,
            zeros_batch_outpn_outchannel,
        ) = layer_info

        in_pc_pad = torch.cat(
            (in_pc, torch.zeros(batch, 1, in_channel).cuda()), 1
        )  # batch*(in_pn+1)*in_channel

        in_neighbors = in_pc_pad[
            :, neighbor_id_lstlst
        ]  # batch*out_pn*max_neighbor_num*in_channel

        ####compute output of convolution layer####
        out_pc_conv = zeros_batch_outpn_outchannel.clone()

        if len(conv_layer) != 0:
            (
                weights,
                bias,
                raw_w_weights,
            ) = conv_layer  # weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num

            w_weights = raw_w_weights * neighbor_mask_lst.view(
                out_pn, max_neighbor_num, 1
            ).repeat(
                1, 1, weight_num
            )  # out_pn*max_neighbor_num*weight_num

            weights = torch.einsum(
                "pmw,wc->pmc", [w_weights, weights]
            )  # out_pn*max_neighbor_num*(out_channel*in_channel)
            weights = weights.view(out_pn, max_neighbor_num, out_channel, in_channel)

            out_neighbors = torch.einsum(
                "pmoi,bpmi->bpmo", [weights, in_neighbors]
            )  # batch*out_pn*max_neighbor_num*out_channel

            out_pc_conv = out_neighbors.sum(2)

            out_pc_conv = out_pc_conv + bias

            if is_final_layer == False:
                out_pc_conv = self.relu(
                    out_pc_conv
                )  ##self.relu is defined in the init function

        ####compute output of residual layer####
        out_pc_res = zeros_batch_outpn_outchannel.clone()

        if len(residual_layer) != 0:
            (
                weight_res,
                p_neighbors_raw,
            ) = residual_layer  # out_channel*in_channel  out_pn*max_neighbor_num

            if in_channel != out_channel:
                in_pc_pad = torch.einsum("oi,bpi->bpo", [weight_res, in_pc_pad])

            out_pc_res = []
            if in_pn == out_pn:
                out_pc_res = in_pc_pad[:, 0:in_pn].clone()
            else:
                in_neighbors = in_pc_pad[
                    :, neighbor_id_lstlst
                ]  # batch*out_pn*max_neighbor_num*out_channel

                p_neighbors = torch.abs(p_neighbors_raw) * neighbor_mask_lst
                p_neighbors_sum = p_neighbors.sum(1) + 1e-8  # out_pn
                p_neighbors = p_neighbors / p_neighbors_sum.view(out_pn, 1).repeat(
                    1, max_neighbor_num
                )

                out_pc_res = torch.einsum("pm,bpmo->bpo", [p_neighbors, in_neighbors])

        # print(out_pc_conv.shape, out_pc_res.shape)
        out_pc = out_pc_conv * np.sqrt(1 - residual_rate) + out_pc_res * np.sqrt(
            residual_rate
        )

        return out_pc

    def forward_till_layer_n(self, in_pc, layer_n):
        out_pc = in_pc.clone()

        for i in range(layer_n):
            if self.test_mode == False:
                out_pc = self.forward_one_conv_layer_batch(out_pc, self.layer_lst[i])
            else:
                out_pc = self.forward_one_conv_layer_batch_during_test(
                    out_pc, self.layer_lst[i]
                )

        # out_pc = self.final_linear(out_pc.transpose(1,2)).transpose(1,2) #batch*3*point_num

        return out_pc

    def forward_from_layer_n(self, in_pc, layer_n):
        out_pc = in_pc.clone()

        for i in range(layer_n, self.layer_num):
            if i < (self.layer_num - 1):
                if self.test_mode == False:
                    out_pc = self.forward_one_conv_layer_batch(
                        out_pc, self.layer_lst[i]
                    )
                else:
                    out_pc = self.forward_one_conv_layer_batch_during_test(
                        out_pc, self.layer_lst[i]
                    )
            else:
                if self.test_mode == False:
                    out_pc = self.forward_one_conv_layer_batch(
                        out_pc, self.layer_lst[i], is_final_layer=True
                    )
                else:
                    out_pc = self.forward_one_conv_layer_batch_during_test(
                        out_pc, self.layer_lst[i], is_final_layer=True
                    )

        return out_pc

    def forward_layer_n(self, in_pc, layer_n):
        out_pc = in_pc.clone()

        if layer_n < (self.layer_num - 1):
            if self.test_mode == False:
                out_pc = self.forward_one_conv_layer_batch(
                    out_pc, self.layer_lst[layer_n]
                )
            else:
                out_pc = self.forward_one_conv_layer_batch_during_test(
                    out_pc, self.layer_lst[layer_n]
                )
        else:
            if self.test_mode == False:
                out_pc = self.forward_one_conv_layer_batch(
                    out_pc, self.layer_lst[layer_n], is_final_layer=True
                )
            else:
                out_pc = self.forward_one_conv_layer_batch_during_test(
                    out_pc, self.layer_lst[layer_n], is_final_layer=True
                )

        return out_pc