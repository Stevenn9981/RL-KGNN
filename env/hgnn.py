import torch.nn as nn
from gym import spaces
from gym.spaces import Discrete
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import *
import torch.nn.functional as F
import collections
import numpy as np
from metrics import *
import time
import torch
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

from KGDataLoader import *

STOP = 0

NEG_SIZE_TRAIN = 10
NEG_SIZE_RANKING = 100


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Net(torch.nn.Module):
    def __init__(self, entity_dim):
        super(Net, self).__init__()
        dout = 0
        self.layer1 = nn.Linear(entity_dim, 32)
        self.layer2 = nn.Linear(32, 64)
        self.conv1 = GATConv(64, 32, 2, dropout=dout)
        self.conv2 = GATConv(64, 64, 2, dropout=dout)
        self.conv3 = GATConv(128, entity_dim, 1, dropout=dout)

    def forward(self, x, edge_index):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.conv1(x, edge_index)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.flatten(x, start_dim=1)
        return x


class hgnn_env(object):
    def __init__(self, logger1, logger2, model_name, dataset='yelp_data', weight_decay=1e-5, policy=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.cur_best = 0
        args = parse_args()
        self.args = args
        # args.data_dir = path
        # print(args.data_dir)
        lr = args.lr
        self.data = DataLoaderHGNN(logger1, args, dataset)
        data = self.data
        # print(data.train_graph)
        # data.train_graph.adj = to_dense_adj(data.train_graph.edge_index, edge_attr=data.train_graph.edge_attr)
        adj_dist = dict()
        attr_dict = dict()
        for i, attr in enumerate(data.train_graph.edge_attr):
            if data.train_graph.edge_index[0][i].item() not in adj_dist:
                adj_dist[data.train_graph.edge_index[0][i].item()] = dict()
            if attr.item() not in adj_dist[data.train_graph.edge_index[0][i].item()]:
                adj_dist[data.train_graph.edge_index[0][i].item()][attr.item()] = set()
            if attr.item() not in attr_dict:
                attr_dict[attr.item()] = set()
            attr_dict[attr.item()].add(data.train_graph.edge_index[0][i].item())
            adj_dist[data.train_graph.edge_index[0][i].item()][attr.item()].add(
                data.train_graph.edge_index[1][i].item())
        data.train_graph.adj_dist = adj_dist
        data.train_graph.attr_dict = attr_dict
        # print(data.train_graph.adj)
        self.model, self.train_data = Net(data.entity_dim).to(self.device), data.train_graph.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        self.train_data.node_idx = self.train_data.node_idx.to(self.device)
        self.data.test_graph = self.data.test_graph.to(self.device)

        self._set_action_space(data.n_relations + 1)
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy
        self.batch_size = args.nd_batch_size
        self.W_R = torch.randn(self.data.n_relations + 1, self.data.entity_dim,
                                             self.data.relation_dim).to(self.device)
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.baseline_experience = 10
        # print(adj_dist)
        # print(data.train_graph.x[random.sample(range(data.train_graph.x.shape[0]), 5)])

        # buffers for updating
        # self.buffers = {i: [] for i in range(max_layer)}
        self.buffers = collections.defaultdict(list)
        self.past_performance = []

        self.meta_path_dict = collections.defaultdict(list)
        self.meta_path_instances_dict = collections.defaultdict(list)
        self.meta_path_graph_edges = collections.defaultdict(set)
        logger1.info('Data initialization done')
        print("finish")

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max)

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        state = self.train_data.x[0]
        self.optimizer.zero_grad()
        return state

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def reset2(self):
        self.meta_path_dict = collections.defaultdict(list)
        self.meta_path_instances_dict = collections.defaultdict(list)
        self.meta_path_graph_edges = collections.defaultdict(set)
        nodes = range(self.train_data.x.shape[0])
        index = random.sample(nodes, min(self.batch_size, len(nodes)))
        state = F.normalize(self.model(self.train_data.x, self.train_data.edge_index)[index]).cpu().detach().numpy()
        self.optimizer.zero_grad()
        return index, state

    def reset2_test(self):
        self.meta_path_dict = collections.defaultdict(list)
        self.meta_path_instances_dict = collections.defaultdict(list)
        nodes = range(self.train_data.x.weight.shape[0])
        index = random.sample(nodes, len(nodes))
        state = F.normalize(self.model(self.train_data.x, self.train_data.edge_index)[index]).cpu().detach().numpy()
        self.optimizer.zero_grad()
        return index, state

    def step2(self, logger1, logger2, index, actions, test=False):
        self.model.train()
        self.optimizer.zero_grad()
        done_list = [False] * self.train_data.x.shape[0]

        next_state, reward, val_acc = [], [], []
        for act, idx in zip(actions, index):
            time1 = time.time()
            logger1.info("Start an iteration")
            if idx not in self.meta_path_graph_edges:
                if act == STOP:
                    self.meta_path_dict[idx].append(STOP)
                    done_list[idx] = True
                elif act not in self.train_data.attr_dict:
                    self.meta_path_dict[idx].append(STOP)
                    pass
                else:
                    self.meta_path_dict[idx].append(act)
                    for start_node in self.train_data.attr_dict[act]:
                        for target_node in self.train_data.adj_dist[start_node][act]:
                            # self.meta_path_instances_dict[idx].append([(start_node, target_node)])
                            self.meta_path_graph_edges[idx].add((start_node, target_node))
            else:
                flag = False
                # update_meta_path_instances = []
                update_meta_path_edges = set()
                if act != STOP and self.meta_path_dict[idx][-1] != STOP:
                    # if len(self.meta_path_instances_dict[idx]) < 2e7 or len(self.meta_path_dict[idx]) < 2:
                    for edge in self.meta_path_graph_edges[idx]:
                        # if len(update_meta_path_instances) > 3e7:
                        #     break
                        end_node = edge[1]
                        if act in self.train_data.adj_dist[end_node]:
                            for target_node in self.train_data.adj_dist[end_node][act]:
                                if target_node != edge[0]:
                                    flag = True
                                    # path_i = path_instance.copy()
                                    # path_i.append((end_node, target_node))
                                    # update_meta_path_instances.append(path_i)
                                    update_meta_path_edges.add((end_node, target_node))
                # self.meta_path_instances_dict[idx] = update_meta_path_instances
                self.meta_path_graph_edges[idx] = update_meta_path_edges
                if flag:
                    self.meta_path_dict[idx].append(act)
                else:
                    if self.meta_path_dict[idx][-1] != STOP:
                        self.meta_path_dict[idx].append(STOP)
                    done_list[idx] = True
            time2 = time.time()
            logger1.info("time2-time1:              %.2f" % (time2 - time1))
            logger1.info("meta-path:                %s" % self.meta_path_dict[idx])
            if test:
                logger2.info("meta-path:                %s" % self.meta_path_dict[idx])
            # logger1.info("meta-path instances: ", self.meta_path_instances_dict[idx])
            # logger1.info("len(meta-path instances): ", len(self.meta_path_instances_dict[idx]))
            logger1.info("len(meta-path edges):     %d" % len(self.meta_path_graph_edges[idx]))

            if len(self.meta_path_graph_edges) > 0 and not done_list[idx]:
                self.train(logger1, idx, test)
                if test:
                    accur = self.test_batch(logger2)
                    if self.cur_best < accur:
                        self.cur_best = accur
                        if os.path.exists(self.model_name):
                            os.remove(self.model_name)
                        torch.save({'state_dict': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'Val': accur,
                                    'Embedding': self.train_data.x},
                                   self.model_name)
                    # self.test_batch(logger2)

            time3 = time.time()
            logger1.info("training time:            %.2f" % (time3 - time2))

            val_precision = self.eval_batch()
            val_acc.append(val_precision)
            self.past_performance.append(val_precision)
            baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
            rew = 100 * (val_precision - baseline)  # FIXME: Reward Engineering
            reward.append(rew)
            logger1.info("Val acc: %.5f  reward: %.5f" % (val_precision, rew))
            logger1.info("-----------------------------------------------------------------------")

        next_state = F.normalize(self.model(self.train_data.x, self.train_data.edge_index)[index]).cpu().detach().numpy()
        r = np.mean(np.array(reward))
        val_acc = np.mean(val_acc)
        next_state = np.array(next_state)

        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'Val': val_acc,
                    'Embedding': self.train_data.x,
                    'Reward': r},
                   'model/epochpoints/e-' + str(val_acc) + '-' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                               time.localtime()) + '.pth.tar')

        logger2.info("Val acc: %.5f  reward: %.5f" % (val_acc, r))

        return next_state, reward, np.array(done_list)[index].tolist(), (val_acc, r)

    def train(self, logger1, idx, test=False):
        self.model.train()
        time1 = time.time()
        edge_index = [[], []]
        edges = set()
        for edge in self.meta_path_graph_edges[idx]:
            edges.add((edge[0], edge[1]))
            edges.add((edge[1], edge[0]))
        logger1.info("len(edges):           %d" % len(edges))
        for edge in edges:
            edge_index[0].append(edge[0])
            edge_index[1].append(edge[1])

        time2 = time.time()
        logger1.info("edge index construction:    %.2f" % (time2 - time1))
        if edge_index == [[], []]:
            return
        # self.train_data.x.weight = nn.Parameter(self.train_data.x.weight.to(self.device))
        edge_index = torch.tensor(edge_index).to(self.device)
        # print(self.data.x.weight.shape)
        # pred = self.model(self.train_data.x(self.train_data.node_idx), edge_index).to(self.device)
        # self.train_data.x = nn.Embedding.from_pretrained(pred, freeze=False)
        # self.train_data.x.weight = nn.Parameter(pred)
        # print(self.train_data.x.weight)

        n_cf_batch = self.data.n_cf_train // self.data.cf_batch_size + 1
        # self.optimizer.zero_grad()

        cf_total_loss = 0
        for iter in range(1, n_cf_batch + 1):
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = self.data.generate_cf_batch(self.data.train_user_dict)
            cf_batch_loss = self.calc_cf_loss(self.train_data, edge_index, cf_batch_user, cf_batch_pos_item,
                                              cf_batch_neg_item, test)
            cf_batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            cf_total_loss += cf_batch_loss

        # cf_total_loss.backward()
        # self.optimizer.step()
        print("total_cf_loss: ", cf_total_loss.item())

        n_kg_batch = self.data.n_kg_train // self.data.kg_batch_size + 1

        # kg_total_loss = 0

        # for iter in range(1, n_kg_batch + 1):
        # kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = self.data.generate_kg_batch(
        #     self.data.train_kg_dict)
        # kg_batch_head = kg_batch_head.to(self.device)
        # kg_batch_relation = kg_batch_relation.to(self.device)
        # kg_batch_pos_tail = kg_batch_pos_tail.to(self.device)
        # kg_batch_neg_tail = kg_batch_neg_tail.to(self.device)
        # kg_batch_loss = self.calc_kg_loss(kg_batch_head, kg_batch_relation, kg_batch_pos_tail,
        #                       kg_batch_neg_tail)
        #
        # kg_batch_loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        # kg_total_loss += kg_batch_loss

        # print("total_kg_loss: ", kg_batch_loss.item())
        # print(self.train_data.x(torch.tensor([10,11,12])))

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.train_data.relation_embed[r]  # (kg_batch_size, relation_dim)

        W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)

        pred = self.model(self.train_data.x, self.train_data.edge_index).to(self.device)

        h_embed = pred[h]  # (kg_batch_size, entity_dim)
        pos_t_embed = pred[pos_t]  # (kg_batch_size, entity_dim)
        neg_t_embed = pred[neg_t]  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def calc_cf_loss(self, g, edge_index, user_ids, item_pos_ids, item_neg_ids, test=False):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """

        pred = self.model(self.train_data.x, edge_index).to(self.device)
        # self.train_data.x.weight = nn.Parameter(pred)
        all_embed = pred  # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]  # (cf_batch_size, cf_concat_dim)
        item_pos_embed = all_embed[item_pos_ids]  # (cf_batch_size, cf_concat_dim)
        item_neg_embed = all_embed[item_neg_ids]  # (cf_batch_size, cf_concat_dim)

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)  # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)  # (cf_batch_size)

        # print("pos, neg: ", pos_score, neg_score)
        # print("user_embedding: ", user_embed)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        # print("cf_loss, l2_loss, loss:", cf_loss.item(), l2_loss.item(), loss.item())
        return loss

    def eval_batch(self, neg_num=NEG_SIZE_TRAIN):
        self.model.eval()
        time1 = time.time()
        user_ids = list(self.data.train_user_dict.keys())
        user_ids_batch = random.sample(user_ids, min(len(user_ids) - 2, self.args.train_batch_size))
        # neg_list = []
        neg_dict = collections.defaultdict(list)
        for u in user_ids_batch:
            for _ in self.data.train_user_dict[u]:
                nl = self.data.sample_neg_items_for_u(self.data.train_user_dict, u, neg_num)
                # neg_list.append(nl)
                neg_dict[u].extend(nl)
        # self.train_data.x.weight = nn.Parameter(self.train_data.x.weight.to(self.device))

        with torch.no_grad():
            all_embed = self.model(self.train_data.x, self.train_data.edge_index).to(self.device)

            time2 = time.time()

            pos_logits = torch.tensor([]).to(self.device)
            neg_logits = torch.tensor([]).to(self.device)

            cf_scores = torch.matmul(all_embed[user_ids_batch],
                                     all_embed[torch.arange(self.data.n_items, dtype=torch.long)].transpose(0, 1))
            for idx, u in enumerate(user_ids_batch):
                pos_logits = torch.cat([pos_logits, cf_scores[idx][self.data.train_user_dict[u]]])
                neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_scores[idx][neg_dict[u]], 1)])
            time3 = time.time()
            NDCG10 = self.metrics(pos_logits, neg_logits).cpu()
            time4 = time.time()
            # print("ALL time: ", time4 - time1)

        return NDCG10.item()

    def test_batch(self, logger2):
        self.model.eval()
        user_ids = list(self.data.test_user_dict.keys())
        user_ids_batch = user_ids[:]

        neg_dict = collections.defaultdict(list)
        NDCG10 = 0

        with torch.no_grad():
            for u in user_ids_batch:
                for _ in self.data.test_user_dict[u]:
                    nl = self.data.sample_neg_items_for_u_test(self.data.train_user_dict, self.data.test_user_dict, u,
                                                               NEG_SIZE_RANKING)
                    neg_dict[u].extend(nl)
            # self.train_data.x.weight = nn.Parameter(self.train_data.x.weight.to(self.device))
            all_embed = self.model(self.train_data.x, self.train_data.edge_index).to(self.device)

            pos_logits = torch.tensor([]).to(self.device)
            neg_logits = torch.tensor([]).to(self.device)

            cf_scores = torch.matmul(all_embed[user_ids_batch],
                                     all_embed[torch.arange(self.data.n_items, dtype=torch.long)].transpose(0, 1))
            for idx, u in enumerate(user_ids_batch):
                pos_logits = torch.cat([pos_logits, cf_scores[idx][self.data.test_user_dict[u]]])
                neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_scores[idx][neg_dict[u]], 1)])

            HR1, HR3, HR10, HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = self.metrics(pos_logits, neg_logits,
                                                                                             training=False)
            logger2.info("HR1 : %.4f, HR3 : %.4f, HR10 : %.4f, HR50 : %.4f, MRR10 : %.4f, MRR20 : %.4f, MRR50 : %.4f, "
                         "NDCG10 : %.4f, NDCG20 : %.4f, NDCG50 : %.4f" % (
                         HR1, HR3, HR10, HR50, MRR10.item(), MRR20.item(),
                         MRR50.item(), NDCG10.item(), NDCG20.item(), NDCG50.item()))
            print("HR1 : %.4f, HR3 : %.4f, HR10 : %.4f, HR50 : %.4f, MRR10 : %.4f, MRR20 : %.4f, MRR50 : %.4f, "
                  "NDCG10 : %.4f, NDCG20 : %.4f, NDCG50 : %.4f" % (HR1, HR3, HR10, HR50, MRR10.item(), MRR20.item(),
                                                                   MRR50.item(), NDCG10.item(), NDCG20.item(),
                                                                   NDCG50.item()))

        return NDCG10.cpu().item()

    def test_train_batch(self):
        self.model.eval()
        user_ids = list(self.data.train_user_dict.keys())
        user_ids_batch = user_ids

        neg_dict = collections.defaultdict(list)
        NDCG10 = 0

        with torch.no_grad():
            for u in user_ids_batch:
                for _ in self.data.train_user_dict[u]:
                    nl = self.data.sample_neg_items_for_u(self.data.train_user_dict, u, NEG_SIZE_RANKING)
                    neg_dict[u].extend(nl)
            # self.train_data.x.weight = nn.Parameter(self.train_data.x.weight.to(self.device))
            all_embed = self.model(self.train_data.x, self.train_data.edge_index).to(self.device)

            pos_logits = torch.tensor([]).to(self.device)
            neg_logits = torch.tensor([]).to(self.device)

            cf_scores = torch.matmul(all_embed[user_ids_batch],
                                     all_embed[torch.arange(self.data.n_items, dtype=torch.long)].transpose(0, 1))
            for idx, u in enumerate(user_ids_batch):
                pos_logits = torch.cat([pos_logits, cf_scores[idx][self.data.train_user_dict[u]]])
                neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_scores[idx][neg_dict[u]], 1)])

            HR1, HR3, HR10, HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = self.metrics(pos_logits, neg_logits,
                                                                                             training=False)
            print("TRAINING DATA: HR1 : %.4f, HR3 : %.4f, HR10 : %.4f, HR50 : %.4f, MRR10 : %.4f, MRR20 : %.4f, "
                  "MRR50 : %.4f, "
                  "NDCG10 : %.4f, NDCG20 : %.4f, NDCG50 : %.4f" % (HR1, HR3, HR10, HR50, MRR10.item(), MRR20.item(),
                                                                   MRR50.item(), NDCG10.item(), NDCG20.item(),
                                                                   NDCG50.item()))

        return NDCG10.cpu().item()


    # def evaluate(self, model, train_graph, train_user_dict, test_user_dict, user_ids_batches, item_ids, K):
    #     model.eval()
    #
    #     n_users = len(test_user_dict.keys())
    #     item_ids_batch = item_ids.cpu().numpy()
    #
    #     cf_scores = []
    #     precision = []
    #     recall = []
    #     # ndcg = []
    #
    #     with torch.no_grad():
    #         for user_ids_batch in user_ids_batches:
    #             cf_scores_batch = self.cf_score(train_graph, user_ids_batch, item_ids)  # (n_batch_users, n_eval_items)
    #             cf_scores_batch = cf_scores_batch.cpu()
    #             user_ids_batch = user_ids_batch.cpu().numpy()
    #             precision_batch, recall_batch = calc_metrics_at_k(cf_scores_batch, train_user_dict,
    #                                                               test_user_dict, user_ids_batch,
    #                                                               item_ids_batch, K)
    #
    #             cf_scores.append(cf_scores_batch.numpy())
    #             precision.append(precision_batch)
    #             recall.append(recall_batch)
    #             # ndcg.append(ndcg_batch)
    #
    #     cf_scores = np.concatenate(cf_scores, axis=0)
    #     precision_k = sum(np.concatenate(precision)) / n_users
    #     recall_k = sum(np.concatenate(recall)) / n_users
    #     # ndcg_k = sum(np.concatenate(ndcg)) / n_users
    #     return cf_scores, precision_k, recall_k  # , ndcg_k

    def cf_score(self, g, user_ids, item_ids):
        """
        user_ids:   number of users to evaluate   (n_eval_users)
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        g.x.weight = nn.Parameter(g.x.weight.to(self.device))
        g = g.to(self.device)
        all_embed = g.x(g.node_idx)  # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]  # (n_eval_users, cf_concat_dim)
        item_embed = all_embed[item_ids]  # (n_eval_items, cf_concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))  # (n_eval_users, n_eval_items)
        return cf_score

    def metrics(self, batch_pos, batch_nega, training=True):
        hit_num1 = 0.0
        hit_num3 = 0.0
        hit_num10 = 0.0
        hit_num50 = 0.0
        mrr_accu10 = torch.tensor(0)
        mrr_accu20 = torch.tensor(0)
        mrr_accu50 = torch.tensor(0)
        ndcg_accu10 = torch.tensor(0)
        ndcg_accu20 = torch.tensor(0)
        ndcg_accu50 = torch.tensor(0)

        if training:
            batch_neg_of_user = torch.split(batch_nega, NEG_SIZE_TRAIN, dim=0)
        else:
            batch_neg_of_user = torch.split(batch_nega, NEG_SIZE_RANKING, dim=0)
        if training:
            for i in range(batch_pos.shape[0]):
                pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0)
                _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
                rank = torch.squeeze((indices == 0).nonzero().to(self.device))
                rank = rank[0]
                if rank < 10:
                    ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
            return ndcg_accu10 / batch_pos.shape[0]
        else:
            for i in range(batch_pos.shape[0]):
                pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0)
                _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
                rank = torch.squeeze((indices == 0).nonzero().to(self.device))
                rank = rank[0]
                if rank < 50:
                    ndcg_accu50 = ndcg_accu50 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    mrr_accu50 = mrr_accu50 + 1 / (rank + 1).type(torch.float32)
                    hit_num50 = hit_num50 + 1
                if rank < 20:
                    ndcg_accu20 = ndcg_accu20 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    mrr_accu20 = mrr_accu20 + 1 / (rank + 1).type(torch.float32)
                if rank < 10:
                    ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    hit_num10 = hit_num10 + 1
                if rank < 10:
                    mrr_accu10 = mrr_accu10 + 1 / (rank + 1).type(torch.float32)
                if rank < 3:
                    hit_num3 = hit_num3 + 1
                if rank < 1:
                    hit_num1 = hit_num1 + 1
            return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num10 / batch_pos.shape[
                0], hit_num50 / \
                   batch_pos.shape[0], mrr_accu10 / batch_pos.shape[0], mrr_accu20 / batch_pos.shape[0], mrr_accu50 / \
                   batch_pos.shape[0], \
                   ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0], ndcg_accu50 / batch_pos.shape[0]
