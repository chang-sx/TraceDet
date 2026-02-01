import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
from utils.predictors.loss import GSATLoss, ConnectLoss, L2Loss
from utils.predictors.loss import Poly1CrossEntropyLoss

from models.encoders.positional_enc import PositionalEncodingTF


class DistributionParams(nn.Module):  
    def __init__(self, input_dim, out_dim, hidden_dim=32):  
        super().__init__()  
        self.fc_mean = MLP([input_dim, hidden_dim, out_dim], activations='elu', dropout=0.0)
        self.fc_logvar = MLP([input_dim, hidden_dim, out_dim], activations='elu', dropout=0.0)
        # self.fc_mean = nn.Linear(input_dim, hidden_dim)  
        # self.fc_logvar = nn.Linear(input_dim, hidden_dim) 
    def forward(self, x):  
        mean = self.fc_mean(x)  
        logvar = self.fc_logvar(x)  
        return mean, logvar  

class TimeHalu(nn.Module):
    def __init__(self, clf, extractor, loss_weight_dict, gsat_r, task, masktoken_stats = None, d_inp=128, d_pe =16):
        super(TimeHalu, self).__init__()
        d_fi = d_inp+d_pe
        self.d_inp = d_inp
        self.task = task
        self.clf = clf
        self.extractor = extractor
        self.device = next(self.parameters()).device
        self.bn = nn.BatchNorm1d(d_fi)
        self.mask_connection_src = MLP([2, 32, 1], activations='elu', dropout=0.0)
        self.loss_weight_dict = loss_weight_dict
        self.gsat_loss_fn = GSATLoss(gsat_r)
        self.L2_loss_fn = L2Loss()
        self.connected_loss = ConnectLoss()
        self.clf_criterion = Poly1CrossEntropyLoss(
            num_classes = 2,
            epsilon = 1.0,
            weight = None,
            reduction = 'mean'
        )
        self.masktoken_stats = masktoken_stats if masktoken_stats is not None else (0.0, 1.0)
        self.mlp = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.ReLU(),
                nn.LayerNorm(d_fi),
                nn.Linear(d_fi, 2),
            )

    def __loss__(self, mask_logits, y_pred_masked, y_pred, Y):
        masked_pred_loss = self.clf_criterion(y_pred_masked, Y)
        orig_pred_loss = self.clf_criterion(y_pred, Y)
        mask_loss = self.loss_weight_dict['gsat'] * self.L2_loss_fn(mask_logits) + self.loss_weight_dict['connect'] * self.connected_loss(mask_logits)

        loss =  0.2*orig_pred_loss + masked_pred_loss + mask_loss 
        return loss, masked_pred_loss, mask_loss


    def forward(self, X, times, Y, captum_input = False):
        '''
        '''
        if self.task == "emb":
            emb = self.clf.embed(X, times, aggregate=False) # [time_length, batch_size, d_model]
            
            mask_prob, rep_mask = self.extractor(emb, X, times)
            _, X_masked = self.multivariate_mask(X, rep_mask)
            X_masked = nn.LayerNorm(X_masked.size(-1)).to(X_masked.device)(X_masked)
            y_pred_masked = self.clf(X_masked, times)
            y_pred = self.clf(X, times)


            
        elif self.task == "entropy":
            emb = self.clf.embed(X, times, aggregate=False) # [time_length, batch_size, d_model]
            mask_prob, rep_mask = self.extractor(emb, X, times)
            ste_mask_rs = mask_prob.transpose(0,1)
            emb_masked = emb * ste_mask_rs

            #--aggrigation
            emb_masked = torch.sum(emb_masked, dim=0) / (torch.sum(ste_mask_rs, dim=0) + 1e-8)
            emb = torch.sum(emb, dim=0) / self.d_inp

            emb = nn.LayerNorm(emb.size(-1)).to(emb.device)(emb)
            emb_masked = nn.LayerNorm(emb_masked.size(-1)).to(emb_masked.device)(emb_masked)

            y_pred = self.mlp(emb)
            y_pred_masked = self.mlp(emb_masked)





        loss, pred_loss, mask_loss = self.__loss__(mask_prob, y_pred_masked, y_pred, Y)
        outputs = {
            "mask_logits": mask_prob,
            "mask_reparam": rep_mask,
            "loss": loss,
            "pred_loss": pred_loss,
            "mask_loss": mask_loss,
            "y_pred_masked": y_pred_masked,
            "y_pred":y_pred 
        }

        return outputs


    def multivariate_mask(self, src, ste_mask):
        # First apply mask directly on input:
        baseline = self._get_baseline(T=src.shape[0], B = src.shape[1], D=src.shape[2])
        
        ste_mask_rs = ste_mask.transpose(0,1)
        if len(ste_mask_rs.shape) == 2:
            ste_mask_rs = ste_mask_rs.unsqueeze(-1)

        src_masked_ref = src * ste_mask_rs # + (1 - ste_mask_rs) * baseline#self.baseline_net(src)#baseline

        
        src_masked = self.mask_connection_src(torch.stack([src, ste_mask_rs.expand(-1, -1, src.size(-1))], dim=-1)).squeeze(-1)

        return src_masked, src_masked_ref
    
    def _get_baseline(self, T, B, D):
        mu, std = self.masktoken_stats 
        mu = mu.unsqueeze(1).expand(T, B, D)  
        std = std.unsqueeze(1).expand(T, B, D)
        samp = torch.normal(mu, std)
        return samp





trans_decoder_default_args = {
    "nhead": 1, 
    "dim_feedforward": 32, 
}

MAX = 10000
class MaskGenerator(nn.Module):
    def __init__(self, 
            d_z, 
            max_len,
            d_pe = 16,
            trend_smoother = False,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            time_net_d_z = 64,
            trans_dec_args = trans_decoder_default_args,
            n_dec_layers = 2,
            tau = 1.0,
            use_ste = True
        ):
        super(MaskGenerator, self).__init__()

        self.d_z = d_z
        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.time_net_d_z = time_net_d_z
        self.agg = agg
        self.max_len = max_len
        self.trend_smoother = trend_smoother
        self.tau = tau
        self.use_ste = use_ste

        self.d_inp = self.d_z - d_pe

        dec_layer = nn.TransformerDecoderLayer(d_model = d_z, **trans_dec_args) 
        self.mask_decoder = nn.TransformerDecoder(dec_layer, num_layers = n_dec_layers)
        
        self.pre_agg_net = nn.Sequential(
            nn.Linear(d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
            nn.Linear(self.pre_agg_mlp_d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
        )

        self.mlp = nn.Sequential(
                nn.Linear(d_z, d_z),
                nn.ReLU(),
                nn.LayerNorm(d_z),
                nn.Linear(d_z, 2),
            )


        self.time_prob_net = nn.Linear(d_z, 2)

        self.ln_in_tgt = nn.LayerNorm(d_z)
        self.ln_in_mem = nn.LayerNorm(d_z)
        self.ln_out = nn.LayerNorm(d_z)


        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        self.init_weights()

    def init_weights(self):
        def iweights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.time_prob_net.apply(iweights)
        self.pre_agg_net.apply(iweights)

    def reparameterize(self, total_mask):


        total_mask_prob = total_mask.softmax(dim=-1)

        total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau = self.tau, hard = self.use_ste)[...,1]

        return total_mask_reparameterize

    def forward(self, z_seq, src, times):
        x = torch.cat([src, self.pos_encoder(times)], dim = -1) # t bs n

        # x = x.transpose(1,0) ###########


        if torch.any(times < -1e5):
            tgt_mask = (times < -1e5).transpose(0,1)
        else:
            tgt_mask = None

    
        
        x =  self.ln_in_tgt(x)
        z_seq_dec = self.mask_decoder(tgt = x, memory = z_seq, tgt_key_padding_mask = tgt_mask)
        p_time = self.time_prob_net(z_seq_dec)

        # p_time = self.mlp(z_seq)


        total_mask_reparameterize = self.reparameterize(p_time.transpose(0,1))
        total_mask = p_time.transpose(0,1).softmax(dim=-1)[...,1].unsqueeze(-1)

        return total_mask, total_mask_reparameterize
        
    def gauss_sample(self, mean_logit, std, training=True):
        if training:
            att_bern = (mean_logit + std * torch.randn(mean_logit.shape, device=mean_logit.device)).sigmoid()
        else:
            att_bern = (mean_logit).sigmoid()
        return att_bern
    
# def kl_divergence_gaussian(mean1, logvar1, mean2, logvar2):  
#     kl_div = 0.5 * torch.mean((logvar2 - logvar1 - 1 + (logvar1.exp() + (mean1 - mean2).pow(2)) / logvar2.exp()))  
#     return kl_div 