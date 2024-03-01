from torch import nn
import torch

class mlp(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.fc1 = torch.nn.Linear(input_size, hidden_size)
    self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
    self.act = torch.nn.ReLU()

  def forward(self, x1, x2):
    h1 = self.fc1(x1)
    h2 = self.fc2(x2)
    # h3 = h1 + h2
    # x = torch.cat([x1, x2], dim=1)  # x (n_node_features, 2*n_node_features)
    h = self.act(h2 + h1)
    return h

class ID(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.fc1 = torch.nn.Linear(input_size, hidden_size)
    # self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
    self.act = torch.nn.ReLU()

  def forward(self, x1, x2):
    # h1 = self.fc1(x1)
    # # h2 = self.fc2(x2)
    # # h3 = h1 + h2
    # # x = torch.cat([x1, x2], dim=1)  # x (n_node_features, 2*n_node_features)
    # h = self.act(h1)
    return x1

class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device, gene):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    check = self.memory.get_last_update(unique_node_ids) <= timestamps
    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device, gene):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device, gene)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device, gene):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device, gene)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)

class MLPMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device, gene):
    super(MLPMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device, gene)

    self.memory_updater = mlp(input_size=message_dimension,
                                     hidden_size=memory_dimension)

class IDMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device, gene):
    super(IDMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device, gene)

    self.memory_updater = ID(input_size=message_dimension,
                                     hidden_size=memory_dimension)
      
def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device, gene):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device, gene)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device, gene)
  elif module_type == "mlp":
    return MLPMemoryUpdater(memory, message_dimension, memory_dimension, device, gene)
  elif module_type == "id":
    return IDMemoryUpdater(memory, message_dimension, memory_dimension, device, gene)
