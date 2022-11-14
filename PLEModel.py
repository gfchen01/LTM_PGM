import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import sys

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class ExpertModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExpertModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TowerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TowerModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        return out

class StackedPLE(nn.Module):
    def __init__(self, input_dim, specific_experts, shared_experts, experts_out, experts_hidden, towers_hidden, tasks_num, PLELayers):
        super(StackedPLE, self).__init__()
        # input_dim, specific_experts, experts_out, experts_hidden, towers_hidden
        self.input_dim = input_dim
        self.share_experts = shared_experts
        self.specific_experts = specific_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks_num = tasks_num

        self.PLELayers = PLELayers



        # 塔和专家
        self.tasks_list = list()

        self.ple_tasks_list = list()

        self.dnn_list = list()
        self.tower_list = list()

        self.specific_experts_list = list()


        self.experts_share = list()


        self.share_gates_list = list()
        self.specific_gates_list = list()

        self.final_experts_share = nn.ModuleList([ExpertModel(experts_out, experts_hidden, experts_out) for i in range(shared_experts)])

        for i in range(tasks_num):
            self.tasks_list.append(nn.ModuleList([ExpertModel(experts_out, experts_hidden, experts_out) for i in range(specific_experts)]))

            self.dnn_list.append(nn.Sequential(nn.Linear(experts_out, shared_experts + specific_experts), nn.Softmax(dim=-1)))
            self.tower_list.append(TowerModel(experts_out, towers_hidden, 1))
            self.add_module('tasks_expert_{}'.format(i + 1), self.tasks_list[i])
            self.add_module('tasks_dnn_{}'.format(i + 1), self.dnn_list[i])
            self.add_module('tasks_tower_{}'.format(i + 1), self.tower_list[i])

        for i in range(PLELayers):
            # self.experts_share.append(nn.ModuleList([ExpertModel(input_dim, experts_hidden, experts_out) for i in range(shared_experts)]))
            if i == 0:
                self.experts_share.append(
                    nn.ModuleList([ExpertModel(input_dim, experts_hidden, experts_out) for i in range(shared_experts)]))
                self.specific_gates_list.append(
                    nn.Sequential(nn.Linear(experts_out * specific_experts, shared_experts + specific_experts), nn.Softmax(dim=-1)))
                self.share_gates_list.append(
                    nn.Sequential(nn.Linear(experts_out * shared_experts, shared_experts + specific_experts * tasks_num), nn.Softmax(dim=-1)))
            else:
                self.experts_share.append(
                    nn.ModuleList([ExpertModel(experts_out, experts_hidden, experts_out) for i in range(shared_experts)]))
                self.specific_gates_list.append(
                    nn.Sequential(nn.Linear(experts_out * specific_experts, shared_experts + specific_experts), nn.Softmax(dim=-1)))
                self.share_gates_list.append(
                    nn.Sequential(nn.Linear(experts_out * shared_experts, shared_experts + specific_experts * tasks_num), nn.Softmax(dim=-1)))
            self.add_module('experts_share_layer_{}'.format(i+1), self.experts_share[i])
            self.add_module('stacked_specific_gates_list_{}'.format(i + 1), self.specific_gates_list[i])
            self.add_module('stacked_share_gates_list_{}'.format(i + 1), self.share_gates_list[i])

        for j in range(PLELayers):
            temp_list = list()
            for i in range(tasks_num):
                if j == 0:
                    temp_list.append(nn.ModuleList([ExpertModel(input_dim, experts_hidden, experts_out) for u in range(specific_experts)]))
                else:
                    temp_list.append(nn.ModuleList([ExpertModel(experts_out, experts_hidden, experts_out) for u in range(specific_experts)]))
                self.add_module('tasks_tower_{}_{}'.format(i + 1, j+1), temp_list[i])
            self.ple_tasks_list.append(temp_list)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def specific_experts_calculation(self, shared_feature, specific_list, specific_gates_list, PLELayer_no, expert_no):
        selected = specific_gates_list[PLELayer_no](specific_list[expert_no].contiguous().view((-1, self.specific_experts * self.experts_out) )).unsqueeze(-1)
        gated_expert_output = torch.cat([shared_feature, specific_list[expert_no]], -1)
        gates_out = torch.bmm(gated_expert_output, selected).squeeze(-1)
        return gates_out

    def gated_select(self, shared_feature, specific_features, specific_list, dnn_list, expert_no):
        selected = dnn_list[expert_no](specific_list[expert_no]).unsqueeze(-1)
        gated_expert_output = torch.cat([shared_feature, specific_features[expert_no]], -1)
        gate_out = torch.bmm(gated_expert_output, selected).squeeze(-1)
        return gate_out, selected

    def share_experts_calculation(self, shared_feature, specific_list, share_gates_list, PLELayer_no):
        selected = share_gates_list[PLELayer_no](shared_feature.reshape(-1, self.share_experts * self.experts_out)).unsqueeze(-1)
        gated_expert_output = torch.cat([shared_feature, torch.cat(specific_list, -1)], -1)
        gates_out = torch.bmm(gated_expert_output, selected).squeeze(-1)
        return gates_out

    def forward(self, x):

        share_features = x
        specific_list = [x for u in range(self.tasks_num)]

        for v in range(self.PLELayers):

            # 计算feature输出
            # 共享专家的结果 tensor(batch_size, hidden_dim)(链表推导式加速运算)
            share_features_temp = torch.cat([e(share_features).unsqueeze(-1) for e in self.experts_share[v]], dim=-1)
            # 第u个专家里面跑第e个任务list[(batch_size, hidden_dim)](链表推导式加速运算)
            specific_experts_temp = [torch.cat([e(specific_list[u]).unsqueeze(-1) for e in self.ple_tasks_list[v][u]], dim=-1) for u in range(self.tasks_num)]

            share_features = self.share_experts_calculation(shared_feature=share_features_temp,
                                                            specific_list=specific_experts_temp,
                                                            share_gates_list=self.share_gates_list,
                                                            PLELayer_no=v)
            # shared_feature, specific_list, specific_gates_list, PLELayer_no, expert_no
            specific_list = [self.specific_experts_calculation(shared_feature=share_features_temp,
                                                               specific_list=specific_experts_temp,
                                                               specific_gates_list=self.specific_gates_list,
                                                                PLELayer_no=v, expert_no=u) for u in range(self.tasks_num)]

        share_features = torch.cat([e(share_features).unsqueeze(-1) for e in self.final_experts_share], dim=-1)
        specific_features = [torch.cat([e(specific_list[u]).unsqueeze(-1) for e in self.tasks_list[u]], dim=-1) for u in range(self.tasks_num)]

        gate_out_tensor = list()
        selected_tensor = list()

        for i in range(self.tasks_num):
            temp_gateout, temp_selected = self.gated_select(share_features, specific_features, specific_list, self.dnn_list, i)
            gate_out_tensor.append(temp_gateout)
            selected_tensor.append(temp_selected)
        selected_tensor = torch.cat(selected_tensor).squeeze(-1)
        final_output = [self.tower_list[u](gate_out_tensor[u]) for u in range(self.tasks_num)]

        return final_output, selected_tensor


class PLEVanilla(nn.Module):
    def __init__(self, input_dim, specific_experts, shared_experts, experts_out, experts_hidden, towers_hidden, tasks_num):
        super(PLEVanilla, self).__init__()
        # input_dim, specific_experts, experts_out, experts_hidden, towers_hidden
        self.input_dim = input_dim
        self.specific_experts = specific_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks_num = tasks_num

        self.experts_share = nn.ModuleList([ExpertModel(input_dim, experts_hidden, experts_out) for i in range(shared_experts)])

        # 塔和专家
        self.tasks_list = list()
        self.dnn_list = list()
        self.tower_list = list()

        for i in range(tasks_num):
            self.tasks_list.append(nn.ModuleList([ExpertModel(input_dim, experts_hidden, experts_out) for i in range(specific_experts)]))
            self.dnn_list.append(nn.Sequential(nn.Linear(input_dim, shared_experts + specific_experts), nn.Softmax(dim=-1)))
            self.tower_list.append(TowerModel(experts_out, towers_hidden, 1))
            self.add_module('tasks_expert_{}'.format(i+1), self.tasks_list[i])
            self.add_module('tasks_dnn_{}'.format(i + 1), self.dnn_list[i])
            self.add_module('tasks_tower_{}'.format(i+1), self.tower_list[i])

    def gated_select(self, input_data, shared_feature, specific_list, dnn_list, expert_no):

        selected = dnn_list[expert_no](input_data).unsqueeze(-1)

        # print('The shapes: ', shared_feature.shape, specific_list[expert_no].shape)

        gated_expert_output = torch.cat([shared_feature, specific_list[expert_no]], -1)
        # print('The shapes: ', selected.shape, gated_expert_output.shape)
        # sys.exit(0)
        gate_out = torch.bmm(gated_expert_output, selected).squeeze(-1)
        return gate_out, selected


    def forward(self, x):
        # 共享专家的结果 tensor(batch_size, hidden_dim)(链表推导式加速运算)
        shared_experts = torch.cat([e(x).unsqueeze(-1) for e in self.experts_share], dim=-1)
        # 第u个专家里面跑第e个任务list[(batch_size, hidden_dim)](链表推导式加速运算)
        specific_experts = [torch.cat([e(x).unsqueeze(-1) for e in self.tasks_list[u]], dim=-1) for u in range(self.tasks_num)]

        gate_out_tensor = list()
        selected_tensor = list()

        for i in range(self.tasks_num):
            temp_gateout, temp_selected = self.gated_select(x, shared_experts, specific_experts, self.dnn_list, i)
            gate_out_tensor.append(temp_gateout)
            selected_tensor.append(temp_selected)
        selected_tensor = torch.cat(selected_tensor).squeeze(-1)
        final_output = [self.tower_list[u](gate_out_tensor[u]) for u in range(self.tasks_num)]

        return final_output, selected_tensor

if __name__ == '__main__':

    DEVICE = torch.device('cuda:0')

    setup_seed(seed=1024)

    test_tensor = torch.randn([64, 7], device=DEVICE)



    print(test_tensor.shape)

    # PLEModelInit = PLEModel(input_dim=20, specific_experts=3, shared_experts=2, experts_out=6, experts_hidden=3, towers_hidden=6, tasks_num=2).to(DEVICE)

    PLEModelInit = StackedPLE(input_dim=7, specific_experts=3, shared_experts=2, experts_out=6, experts_hidden=3,
                              towers_hidden=6, tasks_num=2, PLELayers=2).to(DEVICE)

    final_output, selected_tensor = PLEModelInit(x=test_tensor)

    for i in range(len(final_output)):
        print('The {} th place and shape: {}'.format(i+1, final_output[i].shape))
    print('The selected tensor shape is {}'.format(selected_tensor.shape))
