import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy

# memory module?
# record the node's history in a compressed format
class Memory(nn.Module): # 记录目前所见到的所有节点的状态信息

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum', n_neighbor = 6):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes # number of nodes
    self.memory_dimension = memory_dimension # state's dimension
    self.input_dimension = input_dimension # aggregated messages' dimension
    self.message_dimension = message_dimension # message dimension
    self.device = device # move model to cuda

    self.n_neighbor = n_neighbor

    self.combination_method = combination_method # sum 只出现一次？？？？？？

    self.__init_memory__()

  def __init_memory__(self): # 初始化memory、last update
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False) # memory张量的维度（节点数量*memory维度） 不可传导梯度

    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False) # 与n_nodes维度相同 一维，用于记录上次更新时间

    self.messages = defaultdict(list) # 当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值list:[]，message是一个字典

  def store_raw_messages(self, nodes, node_id_to_messages, aggregator): # Raw Message Store部分

    # zero_temp_inf = [torch.zeros(self.message_dimension).to(self.device) for _ in range(self.n_neighbor)]
    # one_temp_t = [torch.tensor(-1).to(self.device) for _ in range(self.n_neighbor)]
    message_temp = [(torch.zeros(self.message_dimension).to(self.device), torch.tensor(-1).to(self.device)) for _ in range(self.n_neighbor)]
    for node in nodes: # raw message的格式——{源节点id：[源、目的节点的memory，边特征，时间差编码]，[发生时间]}
      if self.messages[node] == [] and aggregator == "attn":
        self.messages[node].extend(message_temp)
        # self.messages_inf[node].extend(zero_temp_inf)
        # self.messages_t[node].extend(one_temp_t)
      # for i in range(len(node_id_to_messages[node])):
      #   info_message = [node_id_to_messages[node][i][0]]
      #   t_message = [node_id_to_messages[node][i][1]]
      self.messages[node].extend(node_id_to_messages[node]) # 提取对应node的raw message，即过去交互的信息
      if aggregator == 'attn':
        self.messages[node] = self.messages[node][-self.n_neighbor:]# only when aggregator is attn
      # self.messages_inf[node].extend(info_message)
      # self.messages_t[node].extend(t_message)




  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :] # （节点数量*维度） 返回第node_idx节点所在行的所有memory信息

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values # 设置node_idx行的具体数值

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs] # 返回第node_idx行上次更新的数据

  def backup_memory(self): # 备份
    messages_clone = {}
    for k, v in self.messages.items(): # 获取message中的item，key与value
      # clone()函数可以返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中。
      messages_clone[k] = [[x[0].clone(), x[1].clone()] for x in v]# set([源、目的节点的memory，边特征，时间差编码]，[发生时间])
    return self.memory.data.clone(), self.last_update.data.clone(), messages_clone # 直接打包成一个大的tensor？

#########这个干啥的？？？？？？？？？？？？？？？
  def restore_memory(self, memory_backup):
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    self.messages = defaultdict(list)
    # self.messages_inf = defaultdict(list)
    # self.messages_t = defaultdict(list)
    for k, v in memory_backup[2].items():
      # new_node_messages = []  # v = time and messsages
      # new_node_messages_inf = []
      # new_node_messages_t = []
      # for message in v:
      #   message_inf = message[0].clone()
      #   message_t = message[1].clone()
      #   new_node_messages.append([message_inf, message_t])  # only detach message information
      #   new_node_messages_inf.append(message_inf)
      #   new_node_messages_t.append(message_t)
      # self.messages[k] = new_node_messages
      # self.messages_t[k] = new_node_messages_t
      # self.messages_inf[k] = new_node_messages_inf
      self.messages[k] = [[x[0].clone(), x[1].clone()] for x in v]

  def detach_memory(self): # 来切断分支的反向传播 共享数据内存且脱离计算图
    self.memory.detach_() # 切断计算图的传播 用于raw_memory

    # Detach all stored messages
    for k, v in self.messages.items():
      new_node_messages = [] # v = time and messsages
      # new_node_messages_inf = []
      # new_node_messages_t = []
      for message in v:
        message_inf = message[0].detach()
        new_node_messages.append([message_inf, message[1]]) # only detach message information
        # self.messages[k].append([message_inf, message[1]])
        # new_node_messages_inf.append(message_inf)
        # # self.messages_inf[k].append(message_inf)
        # new_node_messages_t.append(message[1])
      self.messages[k] = new_node_messages
      # self.messages_inf[k] = new_node_messages_inf
      # self.messages_t[k] = new_node_messages_t

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []
      # self.messages_inf[node] = []
      # self.messages_t[node] = []