import torch
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters*2,kernel_size=k_size, stride=1, padding=k_size//2),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size//2),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size//2),
        )

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 6, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 6, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        # eps = torch.tensor(std.size(), dtype=torch.float32, device='cuda').normal_(0, 0.1)
        # eps = Variable(eps)
        eps = torch.randn_like(std, device='cuda')
        return eps.mul(std).add_(mean)
    

    def forward(self, x):
        x1 = self.conv1(x)
        out, gate = x1.split(int(x1.size(1) / 2), 1)
        x1 = out * torch.sigmoid(gate)
        x2 = self.conv2(x1)
        out, gate = x2.split(int(x2.size(1) / 2), 1)
        x2 = out * torch.sigmoid(gate)
        x3 = self.conv3(x2)
        out, gate = x3.split(int(x3.size(1) / 2), 1)
        x3 = out * torch.sigmoid(gate)
        x1 = nn.functional.interpolate(x1, size=x3.shape[-1], mode='nearest')
        x2 = nn.functional.interpolate(x2, size=x3.shape[-1], mode='nearest')
        
        # Concatenate x1, x2, and x3 along the channel dimension
        concatenated = torch.cat([x1, x2, x3], dim=1)
        
        output = self.out(concatenated)
        
        output = output.squeeze(-1)
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2


class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size,feature_size):
        super(decoder, self).__init__()
        self.feature_size = feature_size
        self.init_dim = init_dim
        self.num_filters = num_filters
        self.k_size = k_size
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3 * (init_dim - 3 * (k_size - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 7, k_size, 1, 0),
            nn.ReLU(),
        )
        #self.fc = nn.Linear(128 * (init_dim - 3 * (k_size - 1) - 3 * (k_size - 1)), feature_size)
        self.fc_initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # defaultDeviceSettings

    def forward(self, x, init_dim, num_filters, k_size):
        x = self.layer(x)

        x = x.view(-1, num_filters * 3, init_dim - 3 * (k_size - 1))

        x = self.convt(x)

        x = x.permute(0, 2, 1)

        x = x.flatten(start_dim=1)  # Flatten the output from conv layers

        #deferInitializationOfSelfFc
        if not self.fc_initialized:
            x_flatten_dim = x.shape[1]
            self.fc = nn.Linear(x_flatten_dim, self.feature_size)
            self.fc.to(self.device)  # move fc to the same device as the input
            self.fc_initialized = True
        
        x = self.fc(x)  # Fully connected layer to match desired output size
        batch_size = x.size(0)  # getBatchSize
        x = x.view(batch_size, -1)  # reshape_x
        return x


class net_reg(nn.Module):
    def __init__(self, num_filters):
        super(net_reg, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

        self.reg1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

        self.reg2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )


    def forward(self, A, B):
        A = self.reg1(A)
        B = self.reg2(B)
        x = torch.cat((A, B), 1)
        x = self.reg(x)
        return x


class net(nn.Module):
    def __init__(self, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2,atom_feature_size, res_feature_size):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(FLAGS.charsmiset_size, 128)
        self.embedding2 = nn.Embedding(FLAGS.charseqset_size, 128)
        self.cnn1 = CNN(NUM_FILTERS, FILTER_LENGTH1)
        self.cnn2 = CNN(NUM_FILTERS, FILTER_LENGTH2)
        self.reg = net_reg(NUM_FILTERS)
        self.decoder1 = decoder(FLAGS.max_smi_len, NUM_FILTERS, FILTER_LENGTH1,FLAGS.charsmiset_size, atom_feature_size)
        self.decoder2 = decoder(FLAGS.max_seq_len, NUM_FILTERS, FILTER_LENGTH2,FLAGS.charseqset_size, res_feature_size)

    def forward(self, x, y, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        x_init = Variable(x.long()).cuda()
        x = self.embedding1(x_init)
        x_embedding = x.permute(0, 2, 1)
        y_init = Variable(y.long()).cuda()
        y = self.embedding2(y_init)
        y_embedding = y.permute(0, 2, 1)
        x, mu_x, logvar_x = self.cnn1(x_embedding)
        y, mu_y, logvar_y = self.cnn2(y_embedding)
        out = self.reg(x, y).squeeze()
        decoded_x = self.decoder1(x, FLAGS.max_smi_len, NUM_FILTERS, FILTER_LENGTH1)
        decoded_y = self.decoder2(y, FLAGS.max_seq_len, NUM_FILTERS, FILTER_LENGTH2)
        return out, decoded_x, decoded_y, x_init, y_init, mu_x, logvar_x, mu_y, logvar_y









