# Import the necessary libraries
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from evaluation.metrics import CoherenceScores
from preprocessing import tokenizer, document_term_matrix, get_dictionary, dataset
import argparse
from utils import gamma_h_boosted, calc_epsilon, my_random_gamma, gamma_h, reparameterize_DIR, reparameterize_GD, kl_DIR, kl_GD, BCE, nll
from torch.distributions.gamma import Gamma
from torch.nn import functional as F
from utils import redundancy, diversity, uniqueness, perplexity_from_loss


#  HeiOnline.csv
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../dataSample.csv', help='the path to the dataset')
parser.add_argument('--vectorizer', type=str, default='tfidf', help='the TfidfVectorizer from sklearn')
#parser.add_argument('--vectorizer', type=str, default='cv', help='the CountVectorizer from sklearn')
parser.add_argument('--min_df', type=int, default=2, help='the minimum number of documents containing a word')
parser.add_argument('--max_df', type=float, default=0.7, help='the maximum number of topics containing a word')
parser.add_argument('--size', type=int, default=100, help='the size of the w2v embeddings')
parser.add_argument('--num_topics', type=int, default=100, help='the number of topics')
parser.add_argument('--top_words', type=int, default=10, help='the number of top words for each topic')
parser.add_argument('--epochs', type=int, default=200, help='the number of the training iterations')
# epoch 200
parser.add_argument('--batch_size', type=int, default=64, help='the size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate of Adam')
parser.add_argument('--b1', type=float, default=0.5, help='the decay of first order momentum of gradient for Adam')
parser.add_argument('--b2', type=float, default=0.999, help='the decay of second order momentum of gradient for Adam')
parser.add_argument('--n_critic', type=int, default=5, help='the number of discriminator iterations per generator_DIR iteration')
parser.add_argument('--hidden_size', type=int, default=100, help="the representation layer's size")
parser.add_argument('--sg', type=int, default=1, help='Training algorithm: 1 for skip-gram, 0 for CBOW.')

opt = parser.parse_args()
CUDA_LAUNCH_BLOCKING=1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prior_alpha = torch.tensor(0.01).to(device)
prior_beta = torch.tensor(0.02).to(device)
prior_alpha_Dir = torch.tensor(0.01).to(device)
# prior_mean = (torch.tensor([0.]*opt.num_topics)).to(device)
#prior_var = (torch.tensor([1. - 1./opt.num_topics]*opt.num_topics)).to(device)

class Encoder_GD(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_topics, batch_size):
        super(Encoder_GD, self).__init__()
        self.batch_size = batch_size

        self.model = nn.Sequential(
        nn.Linear(vocab_size, hidden_size),
        nn.BatchNorm1d(hidden_size, affine=False), # No trainable parameters
        nn.LeakyReLU(0.2, inplace=True),
        #nn.Linear(hidden_size, num_topics),
        #nn.Softmax(1)
        )

        # self.h1 = nn.Linear(hidden_size, num_topics)
        # self.h2 = nn.Linear(hidden_size, num_topics)
        # self.act = nn.Softmax(1)

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        if self.batch_size != batch_size:
            self.batch_size = batch_size

        x = self.model(inputs)
        # alpha = self.act(self.h1(x))
        # beta = self.act(self.h2(x))
        return x #alpha, beta

class Generator_DIR(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_topics, batch_size):
        super(Generator_DIR, self).__init__()
        self.batch_size = batch_size

        self.model = nn.Sequential(
        nn.Linear(vocab_size, hidden_size),
        nn.BatchNorm1d(hidden_size, affine=False), # No trainable parameters
        nn.LeakyReLU(0.2, inplace=True),
        #nn.Linear(hidden_size, num_topics),
        #nn.Softmax(1)
        )
        # self.h1 = nn.Linear(hidden_size, num_topics)
        # self.h2 = nn.Linear(hidden_size, num_topics)
        # self.act = nn.Softmax(1)

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        if self.batch_size != batch_size:
            self.batch_size = batch_size

        x = self.model(inputs)

        # alpha = self.act(self.h1(x))
        # beta = self.act(self.h2(x))
        return x #alpha, beta


class Discriminator(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_topics, batch_size):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size

        self.model = nn.Sequential(
        nn.Linear(num_topics + num_topics, hidden_size),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(hidden_size, vocab_size)
        )


    def forward(self, inputs):
        batch_size = inputs.shape[0]

        if self.batch_size != batch_size:
            self.batch_size = batch_size

        x = self.model(inputs)

        return x


class CoAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CoAttention, self).__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z1, z2):
        Q = self.Wq(z1)  # Query from z1
        K = self.Wk(z2)  # Key from z2
        V = self.Wv(z2)  # Value from z2

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        attention_weights = self.softmax(attention_scores)
        attended_features = torch.matmul(attention_weights, V)
        return attended_features

class fusedFeatures(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics, batch_size):
        super(fusedFeatures, self).__init__()

        self.GD = Encoder_GD(vocab_size, hidden_size, num_topics, batch_size)
        self.Dir = Generator_DIR(vocab_size, hidden_size, num_topics, batch_size)
        self.Desc = Discriminator(vocab_size, hidden_size, num_topics, batch_size)

        self.h1 = nn.Linear(hidden_size, num_topics)
        self.act1 = nn.Softmax(dim=1)
        self.h2 = nn.Linear(hidden_size, num_topics)
        self.act2 = nn.Softmax(dim=1)
        self.h3 = nn.Linear(hidden_size, num_topics)
        self.act3 = nn.Softmax(dim=1)
        
        self.co_attention = CoAttention(num_topics)  # Co-Attention layer

    def reparameterize_GD(self, alpha, beta, num_topics):
        B = 1.0
        alpha = torch.max(torch.tensor(0.0001), alpha)
        beta = torch.max(torch.tensor(0.0001), beta)

        gam1 = torch.squeeze(self.my_random_gamma(shape = (1,), alpha = alpha+ torch.tensor(B)))
        gam2 = torch.squeeze(self.my_random_gamma(shape = (1,), alpha = beta+ torch.tensor(B)))

        eps = (self.calc_epsilon(gam1,alpha+torch.tensor(B))).detach()
        eps2 = (self.calc_epsilon(gam2,beta+torch.tensor(B))).detach()
        u = torch.rand(1,alpha.shape[0],num_topics)
        u2 = torch.rand(1,beta.shape[0],num_topics)
      
        doc_vec = self.gamma_h_boosted(eps,u,alpha,u.shape[0])
        doc_vec2 = self.gamma_h_boosted(eps2,u2,beta,u2.shape[0])

        z = doc_vec/(doc_vec+doc_vec2)

        z = torch.div(doc_vec,torch.reshape(torch.sum(doc_vec,1), (-1, 1)))

        return z
    def gamma_h_boosted(self, epsilon, u, alpha,model_B):
        #(eps,u,alpha,batch_size)
        epsilon = epsilon.to(device)
        u = u.to(device)
        alpha = alpha.to(device)
        
        """
        Reparameterization for gamma rejection sampler with shape augmentation.
        """
        B = u.shape[0]
        K = alpha.shape[1]#(batch_size,K)
        r = torch.arange(0,B) #Me: range(0, n), n=B here
        #Me: reshape r
        rm = (torch.reshape(r,[-1,1,1])).type(torch.FloatTensor).to(device) #dim Bx1x1
        #Me: tile expand and Copy a Tensor: tf.tile(input,multiples,name=None)
        #Me: xy = tf.tile(xs, multiples = [2, 3]) means repeat generate twice in x-ddirection, 3 times in y-direction
        #Me:https://www.tutorialexample.com/understand-tensorflow-tf-tile-expand-a-tensor-tensorflow-tutorial/
        alpha_vec = torch.reshape(torch.tile(alpha,(B,1)),(model_B,-1,K)) + rm #dim BxBSxK + dim Bx1
        alpha_vec = alpha_vec.to(device)
        u_pow = torch.pow(u,1./alpha_vec)+1e-10
        gammah = self.gamma_h(epsilon, alpha + torch.tensor(B))
        return torch.prod(u_pow,axis=0)*gammah


    def calc_epsilon(self, gamma,alpha):
        return torch.sqrt(9.*alpha-3.)*(torch.pow(gamma/(alpha-1./3.),1./3.)-1.)

    def my_random_gamma(self, shape, alpha, beta=1.0):

        alpha = torch.ones(shape).to(device) * alpha
        beta = torch.ones(shape).to(device) * torch.tensor(beta).to(device)

        gamma_distribution = Gamma(alpha, beta)

        return gamma_distribution.sample()

    def gamma_h(self, epsilon, alpha):
        #Me: gamma_h is from equ_2: z=h_gamma(eps, alpha)= EQ2
        """
        Reparameterization for gamma rejection sampler without shape augmentation.
        """
        b = alpha - 1./3.
        c = 1./torch.sqrt(9.*b)
        v = 1.+epsilon*c

        return b*(v**3)

    def reparameterize_DIR(self, alpha, eps=1e-10):
        """
        Reparameterized sampling from a Dirichlet distribution using a Gamma-based trick.

        Args:
            alpha (Tensor): Shape (batch_size, num_classes), concentration parameters of Dirichlet.
            eps (float): Small constant for numerical stability.

        Returns:
            Tensor: Sampled values with shape (batch_size, num_classes)
        """
        alpha = torch.clamp(alpha, min=eps)
        gamma_samples = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).rsample()  # Reparameterized Gamma
        dirichlet_samples = gamma_samples / (gamma_samples.sum(dim=-1, keepdim=True) + eps)  # Normalize
        return dirichlet_samples

    def forward(self, x):
        x1 = self.GD(x)
        x2 = self.Dir(x)

        alpha = self.act1(self.h1(x1))
        beta = self.act2(self.h2(x1))
        z1 = self.reparameterize_GD(alpha, beta, opt.num_topics)

        alpha_Dir = self.act3(self.h3(x2))
        z2 = self.reparameterize_DIR(alpha_Dir)

        z_attended = self.co_attention(z1, z2)  # Apply Co-Attention

        z = torch.cat(
            [(z1 + z_attended) , (z2 + z_attended) ], dim = 1
        )
        #z = z1 + z2 + z_attended  # Fuse the representations

        x_hat = self.Desc(z)

        return x_hat, z, z1, z2, alpha, beta, alpha_Dir


def train_model(model, optimizer, epochs, num_topics, n_critic, device):

    """Return a list of lists each containing the Discriminator's, Generator_DIR's and Encoder_GD's loss, respectively.

        Arguments:

            discriminator: The Discriminator.
            generator_DIR: The Generator_DIR.
            encoder_GD: The Encoder_GD.
            optimizer_d: The optimizer for updating the Discriminator's paratemeters.
            optimizer_g: The optimizer for updating the Generator_DIR's paratemeters.
            optimizer_e: The optimizer for updating the Encoder_GD's paratemeters.
            epochs: The number of the training iterations.
            num_topics: The number of topics.
            n_critic: The number of discriminator iterations per generator_DIR iteration
            device: 'cpu' or 'cuda'.

        Returns:

            train_losses: A list of lists each containing the Discriminator's, Generator_DIR's and Encoder_GD's loss, respectively.

    """


    for epoch in range(epochs):
        losses_d, losses_g, losses_e = [], [], []
        total_losses = []
        total, total_i = 0, 0

        model.train()

        total_num_words = 0
        loss_sum = 0 
        for i, (d_r, _) in enumerate(train_loader):

            model.zero_grad()

            d_r = (d_r/(torch.sum(d_r, 1).unsqueeze(1))).to(device) 
            batch_num_words = torch.sum(d_r, dtype=torch.float32).item()
            total_num_words += batch_num_words
            x_hat, z, z1, z2, alpha, beta, alpha_Dir = model(d_r)
            
            ## Cal KL divergence
            kl_dir_loss = (kl_DIR(alpha_Dir, prior_alpha_Dir,)).mean()
            #print(kl_dir_loss)
            losses_e.append(kl_dir_loss)
            kl_gd_loss =(kl_GD(alpha, beta, prior_alpha, prior_beta, num_topics)).mean()
            #print(kl_gd_loss)
            losses_g.append(kl_gd_loss)

            nll_loss = (nll(x_hat, d_r)).mean()

            loss = kl_dir_loss + kl_gd_loss + nll_loss
            losses_d.append(loss)
            loss_sum += loss.item() 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()


            total = opt.batch_size

            for p in model.parameters():
                p.data.clamp_(-0.01, 0.01) # Clip the weights of the discriminator
                #p = torch.max(p, torch.tensor(0.0001))

        epoch_d = sum(losses_d)/total
        epoch_g = sum(losses_g)/total
        epoch_e = sum(losses_e)/total
        total_losses.append([epoch_d.item(), epoch_g.item(), epoch_e.item()])
        print(f'Epoch {epoch + 1}/{epochs}, Encoder_GD Loss:{epoch_g}, Generator_DIR Loss:{epoch_e}, Discriminator Loss:{epoch_d}')

    return total_losses, x_hat, total_num_words, loss_sum


def get_topics(tfidf, discriminator, num_topics, top_words, vocab_size, device):

    """Returns a list of lists of the top words for each topic.

        Arguments:

            tfidf: The TfidfVectorizer from preprocessing.py.
            model: The Generator_DIR.
            num_topics: The number of topics.
            top_words: The number of the top words for each topics.
            device: 'cpu' or 'cuda'.

        Returns:

            topic_list: A list of lists containing the top words for each topic.

    """

    discriminator.eval()

    with torch.no_grad():
# Generate the topic-word matrix
        onehot_topic = torch.eye((num_topics+num_topics), device=device)
        topic_word_matrix = discriminator(onehot_topic) # discriminator
        #topic_word_matrix, z, z1, z2, alpha, beta, mu, log_var = model(onehot_topic)
# Create a list of lists of the top words for each topic
    topic_list = []

    for topic in topic_word_matrix:
        topic_list.append([tfidf.get_feature_names_out()[j] for j in topic.argsort()[-top_words:]])

# Save the resulted list of lists of words for each topic setting
    df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics+num_topics)])
    df.to_excel(f'BAT_{num_topics}.xlsx')

    return topic_list

def get_doc_topic_list(dtm, model):

    """Return of list of the topic of each document.

        Arguments:

            dtm: An array representing the document term matrix.
            model: The Encoder_GD.

        Returns:

            doc_topic_list: A list of the topics assigned to each document by the Encoder_GD.

    """

    model.eval()
    dtm = torch.FloatTensor(dtm).to(device)

    with torch.no_grad():
         doc_topic_matrix, z, z1, z2, alpha, beta, alpha_Dir = model(dtm)
         doc_topic_list = doc_topic_matrix.argmax(axis=1)

    return doc_topic_list

if __name__ == '__main__':

    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = pd.Series((fetch_20newsgroups(subset='train', shuffle = True)).data)
    newsgroups_test = pd.Series((fetch_20newsgroups(subset='test', shuffle = True)).data)

    # Define the dataset and the arguments
    #df = pd.read_csv(opt.dataset)
    #articles = df['publication_title']

    ##### MANUAL PREPROCESSING ###########################
    #data = []
    #target = []

    # Directory where the dataset is extracted
    #dataset_dir = "../20news-bydate-train"

    #for category in os.listdir(dataset_dir):
    #    category_path = os.path.join(dataset_dir, category)
    #    if os.path.isdir(category_path):
    #        for document in os.listdir(category_path):
    #            document_path = os.path.join(category_path, document)
    #            with open(document_path, "r", errors="ignore") as f:
    #                data.append(f.read())
    #            target.append(category)

    #df = pd.DataFrame({'text': data, 'target': target})

    ##### MANUAL PREPROCESSING ENDS ###########################


    # Generate the document term matrix and the vectorizer
    articles = newsgroups_train.apply(tokenizer) # df['text']

    #processed_articles = SentenceIterator(processed_articles)
    #print(processed_articles)
    tfidf, dtm = document_term_matrix(articles, opt.vectorizer, opt.min_df, opt.max_df)
    # Generate the bag-of-words, the dictionary, and the word2vec model trained on the dataset
    bow, dictionary, w2v = get_dictionary(tfidf, articles, opt.min_df, opt.size, opt.sg)
    # Create the train loader
    train_loader = dataset(dtm, opt.batch_size)

    # Define the models and the optimizers
    vocab_size = dtm.shape[1]
    model = fusedFeatures(vocab_size, opt.hidden_size , opt.num_topics, opt.batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    print("Training starting......................./")
    # Train the model
    total_losses, x_hat, total_num_words, loss_sum = train_model(model, optimizer, opt.epochs, opt.num_topics, opt.n_critic, device)

    # Create the list of lists of the top 10 words of each topic
    discriminator = Discriminator(vocab_size, opt.hidden_size , opt.num_topics, opt.batch_size).to(device)
    topic_list = get_topics(tfidf, discriminator, opt.num_topics, opt.top_words, vocab_size, device)

    # Print the title of the document and its topic based on BAT
    doc_topic_list = get_doc_topic_list(dtm, model)
    #df['Topic'] = doc_topic_list
    #print(df[['title', 'Topic']])

    print('calculate topic redundancy')
    redundancy = redundancy(x_hat, 10)
    print('topic redundancy', str(redundancy))

    print('calculate topic Uniqueness')
    Uniqueness = uniqueness(x_hat, 10)
    print('topic Uniqueness', str(Uniqueness))

    print('calculate topic Diversity')
    Diversity = diversity(x_hat, 10)
    print('topic Diversity', str(Diversity))

    print('Calculating perplexity')
    perplexity = perplexity_from_loss(loss_sum, total_num_words)
    print('topic perplexity', str(perplexity))

    # Calculate the coherence scores
    evaluation_model = CoherenceScores(topic_list, bow, dictionary, w2v)
    coherence_scores = evaluation_model.get_coherence_scores()
    print(coherence_scores)