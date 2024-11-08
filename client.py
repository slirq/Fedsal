import torch


class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args
        
        self.saliency_map=0

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server_se(self, args, server):
        self.gconvNames = server.W.keys()
        if args.alg == 'salstr':
            for k in server.W:
                if '_s' in k:
                    self.W[k].data = server.W[k].data.clone()
        else:
            for k in server.W:
                self.W[k].data = server.W[k].data.clone()

    def download_from_server(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def compute_weight_update(self, local_epoch):
        """ For GCFL """
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device)

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        self.train_stats = train_stats

        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate(self):
        return eval_gc(self.model, self.dataLoader['test'], self.args.device)


    def compute_saliency_maps(self):
        device = self.args.device
        data_loader = self.dataLoader['train']
        model = self.model
        saliency_maps = {name: torch.zeros_like(param).to(device) for name, param in model.named_parameters()}

    
        model.eval()
        size = 0

        for _, batch in enumerate(data_loader):
            batch.to(device)
            model.zero_grad()

            # Ensure batch.x requires gradient
            if batch.x.grad is not None:
                batch.x.grad.zero_()
            batch.x.requires_grad_(True)

            pred = model(batch)
            label = batch.y

            if pred.shape[1] == 1:  # Binary classification, single output (sigmoid output)
                loss = pred[:, 0] if label.item() == 1 else 1 - pred[:, 0]
            else:  # Multi-class classification
                loss = pred[0, label]

            # Compute gradients
            loss = loss.sum()  # Ensure loss is a scalar for backward pass
            loss.backward()

            size += batch.num_graphs

            for name, param in model.named_parameters():
                if param.grad is not None:
                    saliency_maps[name] += param.grad.abs()  # Accumulate absolute gradient values

        # Normalize the saliency maps by the number of samples
        for name in saliency_maps.keys():
            saliency_maps[name] /= size
        
        self.saliency_maps=saliency_maps


    def compute_saliency_norm(client):
        """
        Computes the saliency norm for a client based on its saliency maps.

        Args:
            client (Client): A client object with a `saliency_maps` attribute.

        Returns:
            float: The computed saliency norm.
        """
        saliency_maps = client.saliency_maps

        # Flatten and concatenate saliency maps
        flattened_saliency = []
        for layer_name, saliency_map in saliency_maps.items():
            flattened_saliency.append(saliency_map.view(-1))

        flattened_saliency = torch.cat(flattened_saliency)

        # Compute the norm of the flattened saliency maps
        saliency_norm = torch.norm(flattened_saliency).item()

        return saliency_norm
    
def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])

def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm

def train_gc(model, dataloaders, optimizer, local_epoch, device):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0

        acc_sum = 0

        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_gc(model, val_loader, device)
        loss_tt, acc_tt = eval_gc(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)
    
    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}

def eval_gc(model, test_loader, device):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs

