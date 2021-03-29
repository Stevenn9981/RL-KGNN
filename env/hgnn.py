import torch.nn as nn
from gym import spaces
from gym.spaces import Discrete
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import *
import torch.nn.functional as F
import collections
import numpy as np
from metrics import *

from KGDataLoader import *

STOP = 0

NEG_SIZE_TRAIN = 4
NEG_SIZE_RANKING = 100


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Net(torch.nn.Module):
    def __init__(self, dataset='Cora'):
        super(Net, self).__init__()
        self.conv1 = GATConv(64, 8, heads=8, dropout=0.5)
        self.conv3 = GATConv(64, 8, heads=8, dropout=0.5)
        self.conv2 = GATConv(64, 8, heads=8, dropout=0.5)

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        self.embedding = self.conv2(x, edge_index)
        return self.embedding


class hgnn_env(object):
    def __init__(self, dataset='last-fm', lr=0.01, weight_decay=5e-4, batch_size=128, policy=None):
        self.device = 'cpu'
        # dataset = dataset
        # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        args = parse_args()
        self.args = args
        # args.data_dir = path
        # print(args.data_dir)
        self.data = DataLoaderHGNN(args, dataset)
        data = self.data
        # print(data.train_graph)
        # data.train_graph.adj = to_dense_adj(data.train_graph.edge_index, edge_attr=data.train_graph.edge_attr)
        adj_dist = dict()
        for i, attr in enumerate(data.train_graph.edge_attr):
            if data.train_graph.edge_index[0][i].item() not in adj_dist:
                adj_dist[data.train_graph.edge_index[0][i].item()] = dict()
            if attr.item() not in adj_dist[data.train_graph.edge_index[0][i].item()]:
                adj_dist[data.train_graph.edge_index[0][i].item()][attr.item()] = list()
            adj_dist[data.train_graph.edge_index[0][i].item()][attr.item()].append(
                data.train_graph.edge_index[1][i].item())
        data.train_graph.adj_dist = adj_dist
        # print(data.train_graph.adj)
        self.model, self.train_data = Net(dataset).to(self.device), data.train_graph.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)

        self.data.test_graph = self.data.test_graph.to(self.device)

        self._set_action_space(self.train_data.relation_embed.num_embeddings + 1)
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy
        self.batch_size = args.nd_batch_size
        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.baseline_experience = 50
        # print(adj_dist)
        # print(data.train_graph.x[random.sample(range(data.train_graph.x.shape[0]), 5)])

        # buffers for updating
        # self.buffers = {i: [] for i in range(max_layer)}
        self.buffers = collections.defaultdict(list)
        self.past_performance = []

        self.meta_path_dict = collections.defaultdict(list)
        self.meta_path_instances_dict = collections.defaultdict(list)
        print('Data initialization done')

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max)

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        state = self.train_data.x.weight[0]
        self.optimizer.zero_grad()
        return state

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def reset2(self):
        self.meta_path_dict = collections.defaultdict(list)
        self.meta_path_instances_dict = collections.defaultdict(list)
        nodes = range(self.train_data.x.weight.shape[0])
        index = random.sample(nodes, min(self.batch_size,len(nodes)))
        state = F.normalize(self.train_data.x(torch.tensor(index).to(self.train_data.x.weight.device))).cpu().detach().numpy()
        self.optimizer.zero_grad()
        return index, state

    def step2(self, index, actions):
        self.model.train()
        self.optimizer.zero_grad()
        # start = self.i
        # end = (self.i + self.batch_size) % len(self.train_indexes)
        # index = self.train_indexes[start:end]
        done_list = [False] * self.train_data.x.weight.shape[0]
        # for act, idx in zip(actions, index):
        #     self.buffers[act].append(idx)
        #     if len(self.buffers[act]) >= self.batch_size:
        #         self.train(act, self.buffers[act])
        #         self.buffers[act] = []
        #         done = True
        #     index = self.stochastic_k_hop(actions, index)

        current_state_batch = F.normalize(self.train_data.x(torch.tensor(index)).to(self.device)).detach().numpy()
        next_state, reward, val_acc = [], [], []
        for act, idx in zip(actions, index):
            if idx not in self.meta_path_instances_dict:
                if act == STOP:
                    done_list[idx] = True
                elif act not in self.train_data.adj_dist[idx]:
                    # self.meta_path_instances_dict[idx] = list()
                    pass
                else:
                    self.meta_path_dict[idx].append(act)
                    # print(self.data.adj_dist[idx][act])
                    for target_node in self.train_data.adj_dist[idx][act]:
                        self.meta_path_instances_dict[idx].append([(idx, target_node)])
            else:
                for i in range(len(self.meta_path_instances_dict[idx]) - 1, -1, -1):
                    path_instance = self.meta_path_instances_dict[idx][i]
                    if act == STOP:
                        done_list[idx] = True
                    else:
                        end_node = path_instance[-1][1]
                        if act in self.train_data.adj_dist[end_node]:
                            self.meta_path_dict[idx].append(act)
                            for target_node in self.train_data.adj_dist[end_node][act]:
                                path_instance.append((end_node, target_node))
                        else:
                            self.meta_path_instances_dict[idx].pop(i)
                            if len(self.meta_path_instances_dict[idx]) == 0:
                                del self.meta_path_instances_dict[idx]
            if len(self.meta_path_instances_dict) > 0:
                self.train()
            val_precision = self.eval_batch()
            val_acc.append(val_precision)
            self.past_performance.append(val_precision)
            baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
            rew = 100 * (val_precision - baseline)  # FIXME: Reward Engineering
            reward.append(rew)
            print("Val acc: ", val_precision, " reward: ", rew)
        next_state = F.normalize(self.train_data.x(torch.tensor(index).to(self.train_data.x.weight.device)).cpu()).detach().numpy()
        r = np.mean(np.array(reward))
        val_acc = np.mean(val_acc)
        next_state = np.array(next_state)
        return next_state, reward, np.array(done_list)[index].tolist(), (val_acc, r)

    def train(self):
        self.model.train()
        edge_index = [[], []]
        for paths in self.meta_path_instances_dict.values():
            if len(paths) > 0:
                for path in paths:
                    for edge in path:
                        edge_index[0].append(edge[0])
                        edge_index[1].append(edge[1])
                        edge_index[0].append(edge[1])
                        edge_index[1].append(edge[0])

        if edge_index == [[], []]:
            return
        self.train_data.x.weight = nn.Parameter(self.train_data.x.weight.to(self.device))
        edge_index = torch.tensor(edge_index).to(self.device)
        # print(self.data.x.weight.shape)
        pred = self.model(self.train_data.x(self.train_data.node_idx), edge_index).to(self.device)
        # self.train_data.x = nn.Embedding.from_pretrained(pred, freeze=False)
        self.train_data.x.weight = nn.Parameter(pred)
        cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = self.data.generate_cf_batch(self.data.train_user_dict)
        cf_batch_loss = self.calc_cf_loss(self.train_data, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
        cf_batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def calc_cf_loss(self, g, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = g.x(g.node_idx)                        # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]                            # (cf_batch_size, cf_concat_dim)
        item_pos_embed = all_embed[item_pos_ids]                    # (cf_batch_size, cf_concat_dim)
        item_neg_embed = all_embed[item_neg_ids]                    # (cf_batch_size, cf_concat_dim)

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def eval_batch(self):
        self.model.eval()
        # batch_dict = {}
        # val_index = np.where(self.data.val_mask.to('cpu').numpy()==True)[0]
        # val_states = self.data.x[val_index].to('cpu').numpy()
        # val_acts = self.policy.eval_step(val_states)
        # s_a = zip(val_index, val_acts)
        # for i, a in s_a:
        #     if a not in batch_dict.keys():
        #         batch_dict[a] = []
        #     batch_dict[a].append(i)
        # #acc = 0.0
        # acc = {a: 0.0 for a in range(self.max_layer)}
        # for a in batch_dict.keys():
        #     idx = batch_dict[a]
        #     logits = self.model(a, self.data)
        #     pred = logits[idx].max(1)[1]
        #     #acc += pred.eq(self.data.y[idx]).sum().item() / len(idx)
        #     acc[a] = pred.eq(self.data.y[idx]).sum().item() / len(idx)
        # #acc = acc / len(batch_dict.keys())
        # return acc

        # user_ids = list(self.data.test_user_dict.keys())
        # user_ids_batches = [user_ids[i: i + self.args.test_batch_size] for i in
        #                     range(0, len(user_ids), self.args.test_batch_size)]
        # user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
        # item_ids = torch.arange(self.data.n_items, dtype=torch.long)
        # _, precision, recall = self.evaluate(self.model, self.train_data, self.data.train_user_dict,
        #                                      self.data.test_user_dict,
        #                                      user_ids_batches, item_ids, self.args.K)
        # # print(precision)

        user_ids = list(self.data.train_user_dict.keys())
        user_ids_batch = random.sample(user_ids, self.args.test_batch_size)
        neg_list = [self.data.sample_neg_items_for_u(self.data.train_user_dict, u, NEG_SIZE_TRAIN) for u in user_ids_batch]
        all_embed = self.train_data.x(self.train_data.node_idx)

        pos_logits = torch.tensor([])
        neg_logits = torch.tensor([])

        for idx, u in enumerate(user_ids_batch):
            user_embedding = all_embed[u]
            pos_item_embeddings = all_embed[self.data.train_user_dict[u]]
            cf_score_pos = torch.matmul(user_embedding, pos_item_embeddings.transpose(0, 1))
            neg_item_embeddings = all_embed[neg_list[idx]]
            cf_score_neg = torch.matmul(user_embedding, neg_item_embeddings.transpose(0, 1))\
                .repeat(len(self.data.train_user_dict[u]))
            pos_logits = torch.cat([pos_logits, cf_score_pos])
            neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_score_neg, 1)])

        HR1, HR3, HR20, HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = self.metrics(pos_logits, neg_logits)

        return NDCG10

    def test_batch(self):
        self.model.test()
        user_ids = list(self.data.train_user_dict.keys())
        user_ids_batch = random.sample(user_ids, self.args.test_batch_size)
        neg_list = [self.data.sample_neg_items_for_u(self.data.train_user_dict, u, NEG_SIZE_RANKING) for u in user_ids_batch]
        all_embed = self.train_data.x(self.train_data.node_idx)

        pos_logits = torch.tensor([])
        neg_logits = torch.tensor([])

        for idx, u in enumerate(user_ids_batch):
            user_embedding = all_embed[u]
            pos_item_embeddings = all_embed[self.data.train_user_dict[u]]
            cf_score_pos = torch.matmul(user_embedding, pos_item_embeddings.transpose(0, 1))
            neg_item_embeddings = all_embed[neg_list[idx]]
            cf_score_neg = torch.matmul(user_embedding, neg_item_embeddings.transpose(0, 1))\
                .repeat(len(self.data.train_user_dict[u]))
            pos_logits = torch.cat([pos_logits, cf_score_pos])
            neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_score_neg, 1)])

        HR1, HR3, HR20, HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = self.metrics(pos_logits, neg_logits, training=False)
        print(HR1, HR3, HR20, HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50)

        return NDCG10

    def evaluate(self, model, train_graph, train_user_dict, test_user_dict, user_ids_batches, item_ids, K):
        model.eval()

        n_users = len(test_user_dict.keys())
        item_ids_batch = item_ids.cpu().numpy()

        cf_scores = []
        precision = []
        recall = []
        # ndcg = []

        with torch.no_grad():
            for user_ids_batch in user_ids_batches:
                cf_scores_batch = self.cf_score(train_graph, user_ids_batch, item_ids)  # (n_batch_users, n_eval_items)
                cf_scores_batch = cf_scores_batch.cpu()
                user_ids_batch = user_ids_batch.cpu().numpy()
                precision_batch, recall_batch = calc_metrics_at_k(cf_scores_batch, train_user_dict,
                                                                  test_user_dict, user_ids_batch,
                                                                  item_ids_batch, K)

                cf_scores.append(cf_scores_batch.numpy())
                precision.append(precision_batch)
                recall.append(recall_batch)
                # ndcg.append(ndcg_batch)

        cf_scores = np.concatenate(cf_scores, axis=0)
        precision_k = sum(np.concatenate(precision)) / n_users
        recall_k = sum(np.concatenate(recall)) / n_users
        # ndcg_k = sum(np.concatenate(ndcg)) / n_users
        return cf_scores, precision_k, recall_k  # , ndcg_k

    def cf_score(self, g, user_ids, item_ids):
        """
        user_ids:   number of users to evaluate   (n_eval_users)
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        g.x.weight = nn.Parameter(g.x.weight.to(self.device))
        g = g.to(self.device)
        all_embed = g.x(g.node_idx)           # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]                # (n_eval_users, cf_concat_dim)
        item_embed = all_embed[item_ids]                # (n_eval_items, cf_concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))  # (n_eval_users, n_eval_items)
        return cf_score

    def metrics(self, batch_pos, batch_nega, training=True):
        hit_num1 = 0.0
        hit_num3 = 0.0
        hit_num20 = 0.0
        hit_num50 = 0.0
        mrr_accu10 = 0.0
        mrr_accu20 = 0.0
        mrr_accu50 = 0.0
        ndcg_accu10 = 0.0
        ndcg_accu20 = 0.0
        ndcg_accu50 = 0.0

        if training:
            batch_neg_of_user = torch.split(batch_nega, NEG_SIZE_TRAIN, dim=0)
        else:
            batch_neg_of_user = torch.split(batch_nega, NEG_SIZE_RANKING, dim=0)
        for i in range(batch_pos.shape[0]):
            pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0)
            _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
            rank = torch.squeeze((indices == 0).nonzero().to('cpu'))
            rank = rank[0]
            if rank < 50:
                ndcg_accu50 = ndcg_accu50 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
                mrr_accu50 = mrr_accu50 + 1 / (rank + 1).type(torch.float32)
                hit_num50 = hit_num50 + 1
            if rank < 20:
                ndcg_accu20 = ndcg_accu20 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
                mrr_accu20 = mrr_accu20 + 1 / (rank + 1).type(torch.float32)
                hit_num20 = hit_num20 + 1
            if rank < 10:
                ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
            if rank < 10:
                mrr_accu10 = mrr_accu10 + 1 / (rank + 1).type(torch.float32)
            if rank < 3:
                hit_num3 = hit_num3 + 1
            if rank < 1:
                hit_num1 = hit_num1 + 1
        return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num20 / batch_pos.shape[0], hit_num50 / \
               batch_pos.shape[0], mrr_accu10 / batch_pos.shape[0], mrr_accu20 / batch_pos.shape[0], mrr_accu50 / \
               batch_pos.shape[0], \
               ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0], ndcg_accu50 / batch_pos.shape[0]
