import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
import csv
from pathlib import Path
from decimal import Decimal
from tqdm import  tqdm

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
# from utils.hitrate import hit_prediction
from collections import defaultdict

torch.manual_seed(1)
np.random.seed(1)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')

parser.add_argument('--batch_size', type=int, default=900, help='Batch_size')
parser.add_argument('--n_neighbor', type=int, default=6, help='n_neighbor')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory') # store_true就代表着一旦有这个参数，做出动作“将其值标为True”，
parser.add_argument('--use_age', action='store_true', help='Whether to use age information')

parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn", "mlp", "id"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--Sem', action='store_true', help='Whether to use semantics')
parser.add_argument('--mix', type=str, default="Attn", choices=[
  "Attn", "Sum"], help='Type of semantics aggregator')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

batch_size = args.batch_size
n_neighbor = args.n_neighbor
BATCH_SIZE = args.bs # default = 200
NUM_NEIGHBORS = args.n_degree # default = 10 Number of neighbors to sample
NUM_EPOCH = args.n_epoch # default = 50
NUM_HEADS = args.n_head # default = 2
DROP_OUT = args.drop_out # default = 0.1
GPU = args.gpu
DATA = args.data # default='ç'
NUM_LAYER = args.n_layer # default = 1 Number of network layers
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory # store true
USE_AGE = args.use_age
MESSAGE_DIM = args.message_dim # default = 100
MEMORY_DIM = args.memory_dim # default = 172
aggregator =  args.aggregator
Sem = args.Sem

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
# MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(args.prefix,str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data, node_gene = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, Sem = Sem)


train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
sum_ap = 0
sum_auc = 0
sum_nn_ap = 0
sum_nn_auc = 0
for i in range(args.n_runs):
  MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}--{i}.pth'
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)


  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features, semantic_feature = node_gene,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,use_age = USE_AGE,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
            batch_size=batch_size,
            n_neighbor=n_neighbor, Sem=Sem) # initial the TGN model

  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  # idx_list = np.arange(num_instance)
  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience) # 用于提前结束epoch
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time() # 记录时间
    ### Training
    c_time_list = []
    loss_time_list = []
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    m_num_age = []
    for _ , k in tqdm(enumerate(range(0, num_batch, args.backprop_every))):
      loss = 0
      val_loss = 0
      regula_loss = 0
      regula_loss_pe = 0
      num_age = 0
      optimizer.zero_grad()
      start_time = time.time()
      for j in range(args.backprop_every):


        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)


        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]# *0+1
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        pos_prob, neg_prob, age_num = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, aggregator, NUM_NEIGHBORS)
        num_age += age_num
        c_time_p = time.time() - start_time

        start_time =time.time()
        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
        c_time = time.time()-start_time
        c_time_0 = time.time() - start_epoch
        c_time_list.append(c_time)

      loss /= args.backprop_every
      num_age /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())
      m_num_age.append(num_age)

      loss_time = time.time() - start_time
      loss_time_list.append(loss_time)

      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    if epoch >= 0 or epoch % 10 == 0:
        tgn.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
          train_memory_backup = tgn.memory.backup_memory()

        val_ap, val_auc, val_num_age = eval_edge_prediction(model=tgn,
                                                                negative_edge_sampler=val_rand_sampler,
                                                                data=val_data, aggregator = aggregator,
                                                                n_neighbors=NUM_NEIGHBORS, use_age = USE_AGE)

        if USE_MEMORY:
          val_memory_backup = tgn.memory.backup_memory()
          tgn.memory.restore_memory(train_memory_backup)

        if epoch == None:
            # Validate on unseen nodes
            nn_val_ap, nn_val_auc, nn_num_age = eval_edge_prediction(model=tgn,
                                                        negative_edge_sampler=val_rand_sampler,
                                                        data=new_node_val_data, aggregator = aggregator,
                                                        n_neighbors=NUM_NEIGHBORS, use_age = USE_AGE)
        else:
            nn_val_ap = None
            nn_val_auc = None
            num_age = None
        if USE_MEMORY:
          # Restore memory we had at the end of validation
          tgn.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap) # unseen nodes
        val_aps.append(val_ap)
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
          "val_aps": val_aps, # average precision score
          "new_nodes_val_aps": new_nodes_val_aps,
          "train_losses": train_losses,
          "epoch_times": epoch_times,
          "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('Epoch Val mean loss: {}'.format(val_num_age))
        logger.info('AGE NUMBER: {}'.format(np.mean(m_num_age)))
        logger.info(
          'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        logger.info(
          'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

        # Early stopping
    else:
        val_ap = 0
        val_auc = 0
        num_age = 0
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      flag=1
      break
    else:
      flag = 0
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  if (epoch+1) == NUM_EPOCH and flag != 1:
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    tgn.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch)))
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    tgn.eval()
    flag = 0


  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()


  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc, test_num_age = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data, aggregator = aggregator,
                                                              n_neighbors=NUM_NEIGHBORS, use_age = USE_AGE)


  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  nn_test_ap, nn_test_auc, nn_num_age = eval_edge_prediction(model=tgn,
                                                 negative_edge_sampler=nn_test_rand_sampler,
                                                 data=new_node_test_data, aggregator = aggregator,
                                                 n_neighbors=NUM_NEIGHBORS, use_age = USE_AGE)

  sum_ap += test_ap
  sum_auc += test_auc
  sum_nn_ap += nn_test_ap
  sum_nn_auc += nn_test_auc
  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info('Test Old AGE NUMBER: {}'.format(np.mean(test_num_age)))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
  logger.info('Test New AGE NUMBER: {}'.format(np.mean(nn_num_age)))

  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')

logger.info(
    'Average statistics: Old nodes -- Average auc: {}, Average ap: {}'.format(sum_auc/args.n_runs, sum_ap/args.n_runs))
logger.info(
    'Average statistics: New nodes -- Average auc: {}, Average ap: {}'.format(sum_nn_auc/args.n_runs, sum_nn_ap/args.n_runs))

