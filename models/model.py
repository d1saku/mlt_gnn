import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout, Sequential, ReLU, Conv1d
from torch_geometric.nn import GATConv, TransformerConv, GCNConv, GINConv, global_mean_pool, global_max_pool, global_add_pool


class GNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(GNN, self).__init__() 
        self.transformer_conv1 = TransformerConv(node_feature_dim, 128, edge_dim=edge_feature_dim, heads=4, dropout=dropout_rate)
        self.bn1 = BatchNorm1d(128 * 4) 
        self.transformer_conv2 = TransformerConv(128 * 4, 256, edge_dim=edge_feature_dim, heads=4, dropout=dropout_rate)
        self.bn2 = BatchNorm1d(256 * 4)  
        self.fc_solvent = Linear(solvent_feature_dim, 128)
        self.fc1 = Linear((256 * 4) * 2 + 128, 128)  
        self.bn_fc1 = BatchNorm1d(128)
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(128, output_dim)

    def forward(self, data, solvent_feature_dim=128):
        x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

        x = self.transformer_conv1(x, edge_index, edge_attr)
        x = F.relu(self.bn1(x))
        x = self.transformer_conv2(x, edge_index, edge_attr)
        x = F.relu(self.bn2(x))

        # combination of global mean pooling and global max pooling
        x_gap = global_mean_pool(x, batch)
        x_gmp = global_max_pool(x, batch)  
        x_combined = torch.cat([x_gap, x_gmp], dim=1) 

        # solvent features
        solvent_fingerprint = solvent_fingerprint.view(-1, solvent_feature_dim)
        solvent_features = F.relu(self.fc_solvent(solvent_fingerprint))

        # gnn features + solvent fingerprint
        x = torch.cat([x_combined, solvent_features], dim=1)

        # batchnorm and dropout
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = torch.sigmoid(self.fc2(x))

        return x
    
    
class GCNNet_solvent_graph(torch.nn.Module):
        def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
            super(GCNNet_solvent_graph, self).__init__()

            # Chromophore graphs
            self.gcn_conv1 = GCNConv(node_feature_dim, node_feature_dim)
            self.gcn_conv2 = GCNConv(node_feature_dim, node_feature_dim*2)
            self.gcn_conv3 = GCNConv(node_feature_dim*2, node_feature_dim*4)
            self.fc_g1 = Linear(2*node_feature_dim*4, 1024) 
            self.fc_g2 = Linear(1024, 128)
            self.dropout = Dropout(dropout_rate)
            self.relu = ReLU()

            # Solvent graph
            self.solvent_gcn_conv1 = GCNConv(solvent_feature_dim, solvent_feature_dim)
            self.solvent_gcn_conv2 = GCNConv(solvent_feature_dim, solvent_feature_dim*2)
            self.solvent_gcn_conv3 = GCNConv(solvent_feature_dim*2, solvent_feature_dim*4)
            self.fc_s1 = Linear(2*solvent_feature_dim*4, 1024) 
            self.fc_s2 = Linear(1024, 128)

            # combined features
            self.fc1 = Linear(128+128, 1024)
            self.fc2 = Linear(1024, 512)
            self.out = Linear(512, output_dim)
            

        def forward(self, data, solvent_feature_dim=128):
            x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

            x = self.gcn_conv1(x, edge_index)
            x = self.relu(x)

            x = self.gcn_conv2(x, edge_index)
            x = self.relu(x)
            
            x = self.gcn_conv3(x, edge_index)
            x = self.relu(x)

            # Combination of global mean pooling and global max pooling
            x_gap = global_mean_pool(x, batch)
            x_gmp = global_max_pool(x, batch)
            x_combined = torch.cat([x_gap, x_gmp], dim=1)

            # flattening
            x_combined = self.relu(self.fc_g1(x_combined))
            x_combined = self.dropout(x_combined)
            x_combined = self.relu(self.fc_g2(x_combined))
            x_combined = self.dropout(x_combined)

            # Solvent graph
            solvent_x = self.solvent_gcn_conv1(solvent_fingerprint, edge_index)
            solvent_x = self.relu(solvent_x)

            solvent_x = self.solvent_gcn_conv2(solvent_x, edge_index)
            solvent_x = self.relu(solvent_x)
            
            solvent_x = self.solvent_gcn_conv3(solvent_x, edge_index)
            solvent_x = self.relu(solvent_x)

            # Combination of global mean pooling and global max pooling for solvent graph
            solvent_gap = global_mean_pool(solvent_x, batch)
            solvent_gmp = global_max_pool(solvent_x, batch)
            solvent_combined = torch.cat([solvent_gap, solvent_gmp], dim=1)

            # flattening for solvent graph
            solvent_combined = self.relu(self.fc_s1(solvent_combined))
            solvent_combined = self.dropout(solvent_combined)
            solvent_combined = self.relu(self.fc_s2(solvent_combined))
            solvent_combined = self.dropout(solvent_combined)

            # Combined features
            x = torch.cat([x_combined, solvent_combined], dim=1)

            # Batchnorm and dropout
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.out(x)

            return x
        

class GCNNet(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(GCNNet, self).__init__()

        # Chromophore graphs
        self.gcn_conv1 = GCNConv(node_feature_dim, node_feature_dim)
        self.gcn_conv2 = GCNConv(node_feature_dim, node_feature_dim*2)
        self.gcn_conv3 = GCNConv(node_feature_dim*2, node_feature_dim*4)
        self.fc_g1 = Linear(2*node_feature_dim*4, 1024) 
        self.fc_g2 = Linear(1024, 128)
        self.dropout = Dropout(dropout_rate)
        self.relu = ReLU()

        # Solvent features
        self.fc_solvent = Linear(solvent_feature_dim, 256)
        self.fc_solvent2 = Linear(256, 128)

        # combined features
        self.fc1 = Linear(128+128, 1024)
        self.fc2 = Linear(1024, 512)
        self.out = Linear(512, output_dim)
        

    def forward(self, data, solvent_feature_dim=512):
        x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

        x = self.gcn_conv1(x, edge_index)
        x = self.relu(x)

        x = self.gcn_conv2(x, edge_index)
        x = self.relu(x)
        
        x = self.gcn_conv3(x, edge_index)
        x = self.relu(x)

        # Combination of global mean pooling and global max pooling
        x_gap = global_mean_pool(x, batch)
        x_gmp = global_max_pool(x, batch)
        x_combined = torch.cat([x_gap, x_gmp], dim=1)

        # flattening
        x_combined = self.relu(self.fc_g1(x_combined))
        x_combined = self.dropout(x_combined)
        x_combined = self.relu(self.fc_g2(x_combined))
        x_combined = self.dropout(x_combined)

        # Solvent features
        solvent_fingerprint = solvent_fingerprint.view(-1, solvent_feature_dim)
        solvent_features = self.relu(self.fc_solvent(solvent_fingerprint))
        solvent_features = self.relu(self.fc_solvent2(solvent_features))

        # GNN features + solvent fingerprint
        x = torch.cat([x_combined, solvent_features], dim=1)

        # Batchnorm and dropout
        x = self.dropout(self.relu(self.fc1(x))) # look at the dimensions here
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.out(x)

        return x
    

class GCNNet_conv(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(GCNNet_conv, self).__init__()

        # Chromophore graphs
        self.gcn_conv1 = GCNConv(node_feature_dim, node_feature_dim)
        self.gcn_conv2 = GCNConv(node_feature_dim, node_feature_dim*2)
        self.gcn_conv3 = GCNConv(node_feature_dim*2, node_feature_dim*4)
        self.fc_g1 = Linear(2*node_feature_dim*4, 1024) 
        self.fc_g2 = Linear(1024, 128)
        self.dropout = Dropout(dropout_rate)
        self.relu = ReLU()

        # Solvent features
        self.conv_solvent1 = Conv1d(in_channels=1, out_channels=32, kernel_size=8)
        self.bn_solvent1 = BatchNorm1d(32)
        self.fc_solvent = Linear(505*32, 128)

        # combined features
        self.fc1 = Linear(128+128, 1024)
        self.fc2 = Linear(1024, 512)
        self.out = Linear(512, output_dim)
        

    def forward(self, data, solvent_feature_dim=128):
        x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

        x = self.gcn_conv1(x, edge_index)
        x = self.relu(x)

        x = self.gcn_conv2(x, edge_index)
        x = self.relu(x)
        
        x = self.gcn_conv3(x, edge_index)
        x = self.relu(x)

        # Combination of global mean pooling and global max pooling
        x_gap = global_mean_pool(x, batch)
        x_gmp = global_max_pool(x, batch)
        x_combined = torch.cat([x_gap, x_gmp], dim=1)

        # flattening
        x_combined = self.relu(self.fc_g1(x_combined))
        x_combined = self.dropout(x_combined)
        x_combined = self.relu(self.fc_g2(x_combined))
        x_combined = self.dropout(x_combined)

        # Solvent features
        batch_size = data.batch.max().item() + 1
        solvent_fingerprint = solvent_fingerprint.view(batch_size, 1, -1)
        solvent_features = self.relu(self.bn_solvent1(self.conv_solvent1(solvent_fingerprint)))
        solvent_features = self.relu(self.fc_solvent(solvent_features.view(batch_size, -1)))

        # GNN features + solvent fingerprint
        x = torch.cat([x_combined, solvent_features], dim=1)

        # Batchnorm and dropout
        x = self.dropout(self.relu(self.fc1(x))) # look at the dimensions here
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.out(x)

        return x
    

class GINConvNet(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(GINConvNet, self).__init__()
        
        dim = 32
        self.dropout = Dropout(dropout_rate)
        self.relu = ReLU()

        mlp1 = Sequential(
            Linear(node_feature_dim, dim),
            ReLU(),
            Linear(dim, dim)
        )
        self.gin_conv1 = GINConv(mlp1)
        self.bn1 = BatchNorm1d(dim)

        mlp2 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, dim)
        )
        self.gin_conv2 = GINConv(mlp2)
        self.bn2 = BatchNorm1d(dim)

        mlp3 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, dim)
        )
        self.gin_conv3 = GINConv(mlp3)
        self.bn3 = BatchNorm1d(dim)

        mlp4 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, dim)
        )
        self.gin_conv4 = GINConv(mlp4)
        self.bn4 = BatchNorm1d(dim)

        mlp5 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, dim)
        )
        self.gin_conv5 = GINConv(mlp5)
        self.bn5 = BatchNorm1d(dim)

        self.fc1_graph = Linear(dim, 128)
        
        # Define the fully connected layers for solvent features
        self.fc_solvent = Linear(solvent_feature_dim, 256)
        self.fc_solvent2 = Linear(256, 128)
        
        # Define the fully connected layers for the combined features
        self.fc1 = Linear(128 + 128, 1024)
        self.fc2 = Linear(1024, 256)
        self.out = Linear(256, output_dim)

    def forward(self, data, solvent_feature_dim=512):
        x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

        x = self.gin_conv1(x, edge_index)
        x = self.relu(self.bn1(x))
        x = self.gin_conv2(x, edge_index)
        x = self.relu(self.bn2(x))
        x = self.gin_conv3(x, edge_index)
        x = self.relu(self.bn3(x))
        x = self.gin_conv4(x, edge_index)
        x = self.relu(self.bn4(x))
        x = self.gin_conv5(x, edge_index)
        x = self.relu(self.bn5(x))
        x = global_add_pool(x, batch)
        x = self.relu(self.fc1_graph(x))
        x = F.dropout(x, p=0.3, training=self.training)

        solvent_fingerprint = solvent_fingerprint.view(-1, solvent_feature_dim)
        solvent_features = self.relu(self.fc_solvent(solvent_fingerprint))
        solvent_features = self.relu(self.fc_solvent2(solvent_features))

        x = torch.cat([x, solvent_features], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)

        return x
    

class GAT_GCN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(GAT_GCN, self).__init__()
        
        # Define the GATConv layers
        self.gat_conv1 = GATConv(node_feature_dim, 128, heads=4, dropout=dropout_rate)
        self.bn1 = BatchNorm1d(128 * 4)  # 128 features * 4 heads
        self.gat_conv2 = GATConv(128 * 4, 256, heads=4, dropout=dropout_rate)
        self.bn2 = BatchNorm1d(256 * 4)  # 256 features * 4 heads

        # Define the GCNConv layers
        self.gcn_conv1 = GCNConv(256 * 4, 256)
        self.bn3 = BatchNorm1d(256)

        # Define the fully connected layers for solvent features
        self.fc_solvent = Linear(solvent_feature_dim, 128)
        
        # Define the fully connected layers for the combined features
        self.fc1 = Linear(256 * 2 + 128, 128)  # Adjusted for concatenation of pooled GNN features and solvent features
        self.bn_fc1 = BatchNorm1d(128)
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(128, output_dim)

    def forward(self, data, solvent_feature_dim=128):
        x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

        # GAT layers with batch normalization and ReLU activation
        x = self.gat_conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.gat_conv2(x, edge_index)
        x = F.relu(self.bn2(x))

        # GCN layer with batch normalization and ReLU activation
        x = self.gcn_conv1(x, edge_index)
        x = F.relu(self.bn3(x))

        # Combination of global mean pooling and global max pooling
        x_gap = global_mean_pool(x, batch)
        x_gmp = global_max_pool(x, batch)
        x_combined = torch.cat([x_gap, x_gmp], dim=1)

        # Solvent features processing
        solvent_fingerprint = solvent_fingerprint.view(-1, solvent_feature_dim)
        solvent_features = F.relu(self.fc_solvent(solvent_fingerprint))

        # Concatenate GNN features with solvent features
        x = torch.cat([x_combined, solvent_features], dim=1)

        # Batch normalization, dropout, and final layers
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = torch.sigmoid(self.fc2(x))

        return x