#!/user/bin/python
import random
import collections
import numpy as np

np.random.seed(0)
random.seed(0)


def sample_neg_items_for_u_test(user_dict, test_dict, user_id, n_sample_neg_items):
    pos_items = user_dict[user_id]
    pos_items_2 = test_dict[user_id]

    sample_neg_items = []
    while True:
        if len(sample_neg_items) == n_sample_neg_items:
            break
        neg_item_id = np.random.randint(low=0, high=12677, size=1)[0]
        if neg_item_id not in pos_items and neg_item_id not in pos_items_2 and neg_item_id not in sample_neg_items:
            sample_neg_items.append(neg_item_id)
    return sample_neg_items


train_rate = 0.8

train = []

train_dict = collections.defaultdict(list)
test_dict = collections.defaultdict(list)

with open('data/douban_movie/train.txt', 'r') as infile:
    for line in infile.readlines():
        inter = [int(i) for i in line.strip().split()]
        user, item_ids = inter[0], inter[1:]
        item_ids = list(set(item_ids))
        for item in item_ids:
            train.append([str(user - 24227), str(item + 1), '5'])
            train_dict[user - 24227].append(item + 1)

test = []
with open('data/douban_movie/test.txt', 'r') as infile:
    for line in infile.readlines():
        inter = [int(i) for i in line.strip().split()]
        user, item_ids = inter[0], inter[1:]
        item_ids = list(set(item_ids))
        for item in item_ids:
            test.append([str(user - 24227), str(item + 1), '5'])
            test_dict[user - 24227].append(item + 1)

user_ids_batch = list(test_dict.keys())
neg_dict = collections.defaultdict(list)

for u in user_ids_batch:
    for _ in test_dict[u]:
        nl = sample_neg_items_for_u_test(train_dict, test_dict, u, 5)
        for l in nl:
            train.append([str(u), str(l), '0'])

random.shuffle(train)
random.shuffle(test)

with open('data/douban_movie/ub_' + str(train_rate) + '.train', 'w') as trainfile, \
        open('data/douban_movie/ub_' + str(train_rate) + '.test', 'w') as testfile:
    for r in train:
        trainfile.write('\t'.join(r) + '\n')
    for r in test:
        testfile.write('\t'.join(r) + '\n')
