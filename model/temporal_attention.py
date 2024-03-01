import torch
from torch import nn

from utils.utils import MergeLayer


class TemporalAttentionLayer(torch.nn.Module):
  """
  Just like the Attenmodel in TGAT
  """
  def __init__(self, n_node_features, n_gene, n_neighbors_features, n_edge_features, time_dim,
               output_dimension, n_head=2,
               dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.n_head = n_head
    self.n_gene = n_gene

    self.feat_dim = n_node_features
    self.time_dim = time_dim

    self.query_dim = n_node_features + time_dim + n_gene
    self.key_dim = n_neighbors_features + time_dim + n_edge_features + n_gene

    self.merger = MergeLayer(self.query_dim, n_node_features + n_gene, n_node_features, output_dimension)

    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=n_head,
                                                   dropout=dropout)

  def forward(self, src_node_features, src_time_features, neighbors_features,
              neighbors_time_features, edge_features, neighbors_padding_mask):

    src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)

    query = torch.cat([src_node_features_unrolled, src_time_features], dim=2) # [B, 1, DF + DT]
    # Unlike the query in TGAT, the query here exclude the edge information
    neighbors_time_features = neighbors_time_features.squeeze(dim=1)
    key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2) # [B, N, DF + DE + DT]

    # Reshape tensors so to expected shape by multi head attention
    query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features(DF+DT)]
    key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features(DF+DE+DT)]

    # Compute mask of which source nodes have no valid neighbors
    # neighbors_padding_mask_1 = neighbors_padding_mask
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
    # force the attention to just 'attend' on this neighbor (which has the same features as all
    # the others since they are fake neighbors) and will produce an equivalent result to the
    # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
    # mask保证输入的每个batch即使不等长也不会影响最后的计算
    # print(query.shape, key.shape)

    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                              key_padding_mask=neighbors_padding_mask)

    # mask = torch.unsqueeze(neighbors_padding_mask, dim=2)  # mask [B, N, 1]
    # mask = mask.permute([0, 2, 1])
    # attn_output, attn_output_weights = self.multi_head_target(q=query, k=key, v=key,
    #                                                           mask=mask)

    attn_output = attn_output.squeeze()
    attn_output_weights = attn_output_weights.squeeze() # TGAT中的attn 即α项

    # Source nodes with no neighbors have an all zero attention output. The attention output is
    # then added or concatenated to the original source node features and then fed into an MLP.
    # This means that an all zero vector is not used.
    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

    # Skip connection with temporal attention over neighborhood and the features of the node itself
    # source_time = src_time_features.squeeze(dim = 1)

    attn_output = self.merger(attn_output, src_node_features) # 最后通过一个FFN获得非线性交互信息
    # attn_output = torch.cat((attn_output, source_time), dim = 1)

    return attn_output, attn_output_weights
